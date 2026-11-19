// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible `CAST`.
//!
//! [`spark_cast`] is the kernel, for engines that build physical plans
//! themselves, such as Comet. The `spark_cast(expr, '<type>')` SQL function is
//! built on it.
//!
//! The kernel is being ported from Comet's
//! `native/spark-expr/src/conversion_funcs`, one source type family at a time
//! (<https://github.com/apache/datafusion-spark/issues/88>). A cast with no
//! Spark-compatible implementation yet returns a "not implemented" error, rather
//! than falling through to Arrow's cast and producing a result that differs
//! from Spark's.

mod data_type;
mod numeric;
mod udf;
mod utils;

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray};
use arrow::compute::{CastOptions, cast_with_options};
use arrow::datatypes::{DataType, Int64Type, TimeUnit, TimestampMicrosecondType};
use arrow::temporal_conversions::MICROSECONDS;
use arrow::util::display::FormatOptions;
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, internal_err, not_impl_err,
};
use datafusion_expr::ColumnarValue;

use crate::function::timezone_utils::{local_to_timestamp, timestamp_to_local};
use crate::function::type_coercion::spark_sql_type;
use crate::{EvalMode, SparkTimeZone};
use numeric::{cast_float_to_timestamp, cast_int_to_timestamp};
use utils::remove_trailing_zeroes;

pub(crate) use data_type::parse_spark_type;
pub use udf::SparkCast;

/// The format Spark uses for a timestamp cast to a string. `%.f` writes the
/// fraction of a second only when it is not zero, padded to 3, 6 or 9 digits,
/// and [`remove_trailing_zeroes`] then drops the padding.
const TIMESTAMP_FORMAT: Option<&str> = Some("%Y-%m-%d %H:%M:%S%.f");

/// Options for [`spark_cast`].
///
/// More options, such as the target Spark version, are added as the casts that
/// depend on them are ported.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub struct SparkCastOptions {
    /// How a value that cannot be cast is handled.
    pub eval_mode: EvalMode,
    /// The session time zone (`spark.sql.session.timeZone`). A cast from
    /// `TIMESTAMP` to `TIMESTAMP_NTZ`, `STRING` or `DATE`, or from
    /// `TIMESTAMP_NTZ` to `TIMESTAMP`, returns an error without one.
    pub timezone: Option<SparkTimeZone>,
    /// Why `timezone` is missing, for the error of a cast that needs it.
    missing_timezone_reason: Option<String>,
}

impl SparkCastOptions {
    /// Options with no session time zone.
    pub fn new(eval_mode: EvalMode) -> Self {
        Self {
            eval_mode,
            timezone: None,
            missing_timezone_reason: None,
        }
    }

    pub fn with_timezone(self, timezone: SparkTimeZone) -> Self {
        Self {
            timezone: Some(timezone),
            missing_timezone_reason: None,
            ..self
        }
    }

    /// Options with no session time zone because the configured one could not
    /// be used, for `reason`.
    pub(crate) fn without_timezone_because(self, reason: String) -> Self {
        Self {
            timezone: None,
            missing_timezone_reason: Some(reason),
            ..self
        }
    }

    fn session_timezone(&self) -> Result<&SparkTimeZone> {
        self.timezone
            .as_ref()
            .ok_or_else(|| match &self.missing_timezone_reason {
                Some(reason) => exec_datafusion_err!(
                    "Casting a timestamp requires a session time zone: {reason}"
                ),
                None => exec_datafusion_err!(
                    "Casting a timestamp requires a session time zone"
                ),
            })
    }
}

/// Casts `arg` to `data_type` the way Spark's `Cast` expression does.
///
/// An error is returned for a cast that Spark does not support, or that is not
/// implemented yet. Spark can never name a dictionary type, so a dictionary
/// `data_type` is an internal error.
pub fn spark_cast(
    arg: ColumnarValue,
    data_type: &DataType,
    cast_options: &SparkCastOptions,
) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(cast_array(
            array,
            data_type,
            cast_options,
        )?)),
        ColumnarValue::Scalar(scalar) => {
            let array = cast_array(scalar.to_array()?, data_type, cast_options)?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &array, 0,
            )?))
        }
    }
}

pub(crate) fn cast_array(
    array: ArrayRef,
    to_type: &DataType,
    cast_options: &SparkCastOptions,
) -> Result<ArrayRef> {
    use DataType::*;
    use EvalMode::*;
    let from_type = array.data_type();

    // Spark's SQL type grammar cannot express a dictionary as a cast target:
    // https://github.com/apache/spark/blob/v4.2.0/sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser/SqlBaseParser.g4#L1477-L1525
    if matches!(to_type, Dictionary(_, _)) {
        return internal_err!("Spark cannot specify dictionary types as cast targets");
    }

    if from_type == to_type {
        return Ok(array);
    }

    // Spark's TIMESTAMP and TIMESTAMP_NTZ are microseconds since the epoch, and
    // the timestamp casts below assume that unit.
    let is_other_timestamp_unit =
        |t: &DataType| matches!(t, Timestamp(unit, _) if *unit != TimeUnit::Microsecond);
    if is_other_timestamp_unit(from_type) || is_other_timestamp_unit(to_type) {
        return not_supported(from_type, to_type);
    }

    let eval_mode = cast_options.eval_mode;
    let native_cast_options = CastOptions {
        // Arrow returns NULL for a value it cannot cast when `safe` is set, and
        // an error otherwise.
        safe: eval_mode != Ansi,
        format_options: FormatOptions::new().with_timestamp_format(TIMESTAMP_FORMAT),
    };
    let arrow_cast =
        |array: &dyn Array| Ok(cast_with_options(array, to_type, &native_cast_options)?);

    // Engines such as Comet can read a string or binary Parquet column as a
    // dictionary, although Spark sees its value type. Cast the dictionary values
    // and unpack the result.
    //
    // TODO: this casts values that no row refers to, so once a cast from string
    // can fail, ANSI mode could raise an error that Spark would not.
    if let Dictionary(_, value_type) = from_type
        && matches!(value_type.as_ref(), Utf8 | LargeUtf8 | Binary | LargeBinary)
    {
        let dictionary = array.as_any_dictionary();
        let values = cast_array(Arc::clone(dictionary.values()), to_type, cast_options)?;
        return arrow_cast(&dictionary.with_values(values));
    }

    match (from_type, to_type) {
        // A NULL array has no values, so Arrow's cast only changes the type.
        (Null, _) => arrow_cast(&array),

        // Numbers are seconds since the epoch. Spark has no cast from a number
        // to TIMESTAMP_NTZ.
        (Int8 | Int16 | Int32 | Int64, Timestamp(_, tz @ Some(_))) => {
            cast_int_to_timestamp(&array, tz)
        }
        (Float32 | Float64, Timestamp(_, tz @ Some(_))) => {
            cast_float_to_timestamp(&array, tz, eval_mode)
        }

        // Timestamps. Arrow's casts ignore the session time zone.
        (Timestamp(_, Some(_)), Int64) => {
            // Spark returns whole seconds, rounded down.
            let micros = array.as_primitive::<TimestampMicrosecondType>();
            Ok(Arc::new(
                micros.unary::<_, Int64Type>(|v| v.div_euclid(MICROSECONDS)),
            ))
        }
        (Timestamp(_, Some(_)), Utf8 | Date32 | Timestamp(_, None)) => {
            let local = timestamp_to_local(
                array.as_primitive(),
                cast_options.session_timezone()?,
            )?;
            match to_type {
                Utf8 => Ok(remove_trailing_zeroes(&arrow_cast(&local)?)),
                Date32 => arrow_cast(&local),
                _ => Ok(Arc::new(local)),
            }
        }
        (Timestamp(_, None), Utf8) => Ok(remove_trailing_zeroes(&arrow_cast(&array)?)),
        (Timestamp(_, None), Timestamp(_, Some(target_tz))) => {
            let utc = local_to_timestamp(
                array.as_primitive(),
                cast_options.session_timezone()?,
            )?;
            Ok(Arc::new(utc.with_timezone(Arc::clone(target_tz))))
        }

        // Casts that Arrow supports, but where Spark's result differs, and that
        // have not been ported from Comet yet. Each is replaced by its
        // implementation as its family is ported.
        //
        // LEGACY wraps on overflow and ANSI raises CAST_OVERFLOW, while Arrow
        // returns NULL. TRY matches Arrow.
        (Int64, Int32 | Int16 | Int8) | (Int32, Int16 | Int8) | (Int16, Int8)
            if eval_mode != Try =>
        {
            not_yet_ported(from_type, to_type, eval_mode)
        }
        (Float32 | Float64 | Decimal128(_, _), Int8 | Int16 | Int32 | Int64)
            if eval_mode != Try =>
        {
            not_yet_ported(from_type, to_type, eval_mode)
        }
        // ANSI raises NUMERIC_VALUE_OUT_OF_RANGE, with Spark's message.
        (Decimal128(_, _), Decimal128(_, _)) if eval_mode == Ansi => {
            not_yet_ported(from_type, to_type, eval_mode)
        }
        // LEGACY uses Java's BigDecimal.toString, as in "0E-18".
        (Decimal128(_, _), Utf8) if eval_mode == Legacy => {
            not_yet_ported(from_type, to_type, eval_mode)
        }
        // Spark replaces invalid UTF-8, and Arrow raises an error.
        (Binary, Utf8) => not_yet_ported(from_type, to_type, eval_mode),
        // Casts Spark supports that Arrow does not do the same way.
        (
            Utf8 | LargeUtf8 | Utf8View,
            Boolean
            | Int8
            | Int16
            | Int32
            | Int64
            | Float32
            | Float64
            | Decimal128(_, _)
            | Date32
            | Timestamp(_, _),
        )
        | (Boolean | Decimal128(_, _), Timestamp(_, Some(_)))
        | (Boolean | Int8 | Int16 | Int32 | Int64, Decimal128(_, _))
        | (Int8 | Int16 | Int32 | Int64, Binary)
        | (Float32 | Float64, Decimal128(_, _) | Utf8)
        | (Decimal128(_, _), Boolean | Float32 | Float64)
        | (Date32, Timestamp(_, _))
        | (
            Timestamp(_, Some(_)),
            Boolean | Int8 | Int16 | Int32 | Float32 | Float64 | Decimal128(_, _),
        )
        | (
            Date32,
            Boolean | Int8 | Int16 | Int64 | Float32 | Float64 | Decimal128(_, _),
        )
        | (Struct(_) | List(_) | Map(_, _), Utf8)
        | (Struct(_), Struct(_))
        | (List(_), List(_))
        | (Map(_, _), Map(_, _)) => not_yet_ported(from_type, to_type, eval_mode),
        // LEGACY returns NULL and ANSI rejects the cast, while Arrow reinterprets
        // the days since the epoch.
        (Date32, Int32) => not_yet_ported(from_type, to_type, eval_mode),

        _ if is_datafusion_spark_compatible(from_type, to_type) => arrow_cast(&array),
        _ => not_supported(from_type, to_type),
    }
}

fn not_supported(from_type: &DataType, to_type: &DataType) -> Result<ArrayRef> {
    not_impl_err!(
        "Spark-compatible cast from {} to {} is not supported",
        spark_sql_type(from_type),
        spark_sql_type(to_type)
    )
}

fn not_yet_ported(
    from_type: &DataType,
    to_type: &DataType,
    eval_mode: EvalMode,
) -> Result<ArrayRef> {
    not_impl_err!(
        "Spark-compatible cast from {} to {} in {eval_mode:?} mode is not implemented yet",
        spark_sql_type(from_type),
        spark_sql_type(to_type)
    )
}

/// Whether Arrow's cast from `from_type` to `to_type` matches Spark's in every
/// eval mode. [`cast_array`] handles the casts it needs to before consulting
/// this.
fn is_datafusion_spark_compatible(from_type: &DataType, to_type: &DataType) -> bool {
    use DataType::*;
    match from_type {
        Boolean => matches!(
            to_type,
            Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Utf8
        ),
        Int8 | Int16 | Int32 | Int64 => matches!(
            to_type,
            Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64 | Utf8
        ),
        Float32 | Float64 => matches!(
            to_type,
            Boolean | Int8 | Int16 | Int32 | Int64 | Float32 | Float64
        ),
        // Float32 and Float64 are absent: Arrow computes
        // `(unscaled as f64) / 10^scale`, which rounds twice and can differ from
        // Spark's correctly rounded BigDecimal.doubleValue by one ulp. Boolean is
        // absent because Arrow does not implement that cast.
        Decimal128(_, _) | Decimal256(_, _) => matches!(
            to_type,
            Int8 | Int16 | Int32 | Int64 | Decimal128(_, _) | Decimal256(_, _) | Utf8
        ),
        Utf8 => matches!(to_type, Binary),
        Date32 => matches!(to_type, Utf8),
        // The instant is unchanged, so Arrow only replaces the time zone.
        Timestamp(_, Some(_)) => matches!(to_type, Timestamp(_, Some(_))),
        Timestamp(_, None) => matches!(to_type, Date32),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::TimestampMillisecondArray;
    use arrow::array::{
        BinaryArray, DictionaryArray, Int32Array, Int64Array, NullArray, StringArray,
        TimestampMicrosecondArray,
    };
    use arrow::datatypes::{Field, Int32Type, Int64Type};
    use datafusion_common::DataFusionError;

    fn options(eval_mode: EvalMode, timezone: &str) -> SparkCastOptions {
        SparkCastOptions::new(eval_mode)
            .with_timezone(SparkTimeZone::parse(timezone).unwrap())
    }

    fn legacy() -> SparkCastOptions {
        options(EvalMode::Legacy, "UTC")
    }

    fn timestamp(tz: Option<&str>) -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, tz.map(Into::into))
    }

    #[test]
    fn test_cast_to_dictionary_is_rejected() {
        let error = cast_array(
            Arc::new(StringArray::from(vec!["a"])),
            &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            &legacy(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Spark cannot specify dictionary types as cast targets"),
            "{error}"
        );
    }

    #[test]
    fn test_cast_to_same_type_returns_input() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let result = cast_array(Arc::clone(&array), &DataType::Int32, &legacy()).unwrap();
        assert!(Arc::ptr_eq(&array, &result));
    }

    #[test]
    fn test_unsupported_cast_is_an_error() {
        // Spark has no cast from DATE to BINARY.
        let error = cast_array(
            Arc::new(arrow::array::Date32Array::from(vec![1])),
            &DataType::Binary,
            &legacy(),
        )
        .unwrap_err();
        assert!(
            matches!(error, DataFusionError::NotImplemented(_)),
            "{error}"
        );
        assert!(
            error
                .to_string()
                .contains("cast from DATE to BINARY is not supported"),
            "{error}"
        );

        // Spark supports a cast from STRING to INT, which is not ported yet.
        let error = cast_array(
            Arc::new(StringArray::from(vec!["1"])),
            &DataType::Int32,
            &legacy(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("not implemented yet"), "{error}");
    }

    /// Casts from TIMESTAMP and DATE to numbers and BOOLEAN exist in Spark, so
    /// they are reported as not implemented rather than not supported.
    #[test]
    fn test_timestamp_and_date_to_numbers_are_pending() {
        let timestamp: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC"));
        let date: ArrayRef = Arc::new(arrow::array::Date32Array::from(vec![0]));
        let targets = [
            DataType::Boolean,
            DataType::Int16,
            DataType::Float64,
            DataType::Decimal128(20, 2),
        ];
        for (array, target) in [&timestamp, &date]
            .into_iter()
            .flat_map(|array| targets.iter().map(move |t| (array, t)))
        {
            let error = cast_array(Arc::clone(array), target, &legacy()).unwrap_err();
            assert!(error.to_string().contains("not implemented yet"), "{error}");
        }
    }

    /// Arrow reinterprets the days since the epoch, while Spark returns NULL
    /// (LEGACY) or rejects the cast (ANSI), so DATE to INT is not ported yet.
    #[test]
    fn test_date_to_int_is_not_reinterpreted() {
        let array: ArrayRef = Arc::new(arrow::array::Date32Array::from(vec![18262]));
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi, EvalMode::Try] {
            let error = cast_array(
                Arc::clone(&array),
                &DataType::Int32,
                &SparkCastOptions::new(eval_mode),
            )
            .unwrap_err();
            assert!(error.to_string().contains("not implemented yet"), "{error}");
        }
    }

    /// A cast whose Spark-compatible implementation has not been ported must
    /// not fall through to Arrow's cast, whose result differs from Spark's.
    #[test]
    fn test_pending_cast_does_not_fall_through_to_arrow() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![i64::from(i32::MAX) + 1]));
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi] {
            let error = cast_array(
                Arc::clone(&array),
                &DataType::Int32,
                &SparkCastOptions::new(eval_mode),
            )
            .unwrap_err();
            assert!(
                matches!(error, DataFusionError::NotImplemented(_)),
                "{error}"
            );
            assert!(error.to_string().contains("not implemented yet"), "{error}");
        }

        // TRY matches Arrow's safe cast, which returns NULL on overflow.
        let result = cast_array(
            array,
            &DataType::Int32,
            &SparkCastOptions::new(EvalMode::Try),
        )
        .unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn test_arrow_fallthrough() {
        // A widening cast cannot fail in any mode.
        for eval_mode in [EvalMode::Legacy, EvalMode::Ansi, EvalMode::Try] {
            let result = cast_array(
                Arc::new(Int32Array::from(vec![Some(-1), None])),
                &DataType::Int64,
                &SparkCastOptions::new(eval_mode),
            )
            .unwrap();
            assert_eq!(
                result.as_primitive::<Int64Type>(),
                &Int64Array::from(vec![Some(-1), None])
            );
        }

        let result = cast_array(
            Arc::new(Int32Array::from(vec![Some(0), Some(7), None])),
            &DataType::Boolean,
            &legacy(),
        )
        .unwrap();
        let result = result.as_boolean();
        assert!(!result.value(0) && result.value(1) && result.is_null(2));
    }

    #[test]
    fn test_scalar() {
        let result = spark_cast(
            ColumnarValue::Scalar(ScalarValue::Int32(Some(42))),
            &DataType::Utf8,
            &legacy(),
        )
        .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert_eq!(scalar, ScalarValue::Utf8(Some("42".to_string())));

        let result = spark_cast(
            ColumnarValue::Scalar(ScalarValue::Int32(None)),
            &DataType::Int64,
            &legacy(),
        )
        .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert_eq!(scalar, ScalarValue::Int64(None));
    }

    #[test]
    fn test_cast_null_array() {
        for to_type in [
            DataType::Int32,
            DataType::Utf8,
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
        ] {
            let result =
                cast_array(Arc::new(NullArray::new(3)), &to_type, &legacy()).unwrap();
            assert_eq!(result.data_type(), &to_type);
            assert_eq!(result.null_count(), 3);
        }
    }

    #[test]
    fn test_cast_timestamp_to_long_rounds_down() {
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![1_500_000, -1_500_000])
                .with_timezone("UTC"),
        );
        let result = cast_array(array, &DataType::Int64, &legacy()).unwrap();
        assert_eq!(
            result.as_primitive::<Int64Type>(),
            &Int64Array::from(vec![1, -2])
        );
    }

    #[test]
    fn test_cast_timestamp_to_string_in_session_timezone() {
        // 2024-01-15 15:30:00.1 UTC
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(1_705_332_600_100_000), None])
                .with_timezone("UTC"),
        );
        let options = options(EvalMode::Legacy, "America/New_York");
        let result = cast_array(array, &DataType::Utf8, &options).unwrap();
        assert_eq!(
            result.as_string::<i32>(),
            &StringArray::from(vec![Some("2024-01-15 10:30:00.1"), None])
        );
    }

    #[test]
    fn test_cast_timestamp_to_date_in_session_timezone() {
        // 2024-01-15 03:00:00 UTC is still 2024-01-14 in Los Angeles.
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![1_705_287_600_000_000])
                .with_timezone("UTC"),
        );
        let result =
            cast_array(array, &DataType::Date32, &options(EvalMode::Legacy, "PST"))
                .unwrap();
        assert_eq!(
            result
                .as_primitive::<arrow::datatypes::Date32Type>()
                .value(0),
            19_736
        );
    }

    #[test]
    fn test_cast_between_timestamp_and_timestamp_ntz() {
        // 2024-01-15 15:30:00 UTC is 10:30:00 in New York.
        let utc_micros = 1_705_332_600_000_000;
        let local_micros = 1_705_314_600_000_000;
        let options = options(EvalMode::Legacy, "America/New_York");

        let ltz: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![utc_micros]).with_timezone("UTC"),
        );
        let ntz = cast_array(ltz, &timestamp(None), &options).unwrap();
        assert_eq!(ntz.data_type(), &timestamp(None));
        assert_eq!(
            ntz.as_primitive::<TimestampMicrosecondType>().value(0),
            local_micros
        );

        let ltz = cast_array(ntz, &timestamp(Some("UTC")), &options).unwrap();
        assert_eq!(ltz.data_type(), &timestamp(Some("UTC")));
        assert_eq!(
            ltz.as_primitive::<TimestampMicrosecondType>().value(0),
            utc_micros
        );
    }

    #[test]
    fn test_cast_timestamp_in_other_units_is_unsupported() {
        let millis: ArrayRef = Arc::new(TimestampMillisecondArray::from(vec![1_500]));
        let error = cast_array(millis, &DataType::Int64, &legacy()).unwrap_err();
        assert!(
            matches!(error, DataFusionError::NotImplemented(_)),
            "{error}"
        );

        let micros: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![0]));
        let nanos = DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into()));
        let error = cast_array(micros, &nanos, &legacy()).unwrap_err();
        assert!(
            matches!(error, DataFusionError::NotImplemented(_)),
            "{error}"
        );
    }

    #[test]
    fn test_cast_timestamp_without_timezone_is_an_error() {
        let array: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC"));
        let options = SparkCastOptions::new(EvalMode::Legacy);
        let error = cast_array(array, &DataType::Utf8, &options).unwrap_err();
        assert!(
            error.to_string().contains("requires a session time zone"),
            "{error}"
        );
    }

    #[test]
    fn test_cast_timestamp_ntz() {
        // 2024-01-15 10:30:00.5 local time
        let ntz: ArrayRef =
            Arc::new(TimestampMicrosecondArray::from(vec![1_705_314_600_500_000]));
        // A local date-time renders without the session time zone.
        let options = SparkCastOptions::new(EvalMode::Legacy);
        let result = cast_array(Arc::clone(&ntz), &DataType::Utf8, &options).unwrap();
        assert_eq!(
            result.as_string::<i32>(),
            &StringArray::from(vec!["2024-01-15 10:30:00.5"])
        );
        let error = cast_array(ntz, &DataType::Int64, &legacy()).unwrap_err();
        assert!(
            matches!(error, DataFusionError::NotImplemented(_)),
            "{error}"
        );
    }

    #[test]
    fn test_cast_unsupported_timestamp_to_date() {
        // chrono cannot represent every value of a microsecond timestamp.
        let array: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![i64::MAX]).with_timezone("UTC"),
        );
        let options = options(EvalMode::Legacy, "Europe/Copenhagen");
        assert!(cast_array(array, &DataType::Date32, &options).is_err());
    }

    #[test]
    fn test_cast_dictionary_string_unpacks_result() {
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![Some(1), None, Some(0), Some(1)]),
                Arc::new(StringArray::from(vec!["a", "bc"])),
            )
            .unwrap(),
        );
        let result = cast_array(dictionary, &DataType::Binary, &legacy()).unwrap();
        assert_eq!(
            result.as_binary::<i32>(),
            &BinaryArray::from(vec![
                Some(b"bc".as_slice()),
                None,
                Some(b"a".as_slice()),
                Some(b"bc".as_slice()),
            ])
        );
    }

    #[test]
    fn test_cast_dictionary_of_other_types_is_unsupported() {
        let dictionary: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![0]),
                Arc::new(Int32Array::from(vec![1])),
            )
            .unwrap(),
        );
        assert!(cast_array(dictionary, &DataType::Int64, &legacy()).is_err());
    }
}
