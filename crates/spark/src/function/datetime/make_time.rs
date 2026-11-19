// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, PrimitiveArray, Time64NanosecondArray};
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::{DataType, Decimal128Type, Int32Type, TimeUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, exec_err, internal_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;

const MICROS_PER_SECOND: i128 = 1_000_000;
const NANOS_PER_MICRO: i64 = 1_000;

/// Spark's seconds argument type, `DecimalType(16, 6)`.
const SECONDS_TYPE: DataType = DataType::Decimal128(16, 6);

/// Spark-compatible `make_time` expression (Spark 4.1+).
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#make_time>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Creates a time of day from hour, minute and second fields. The result is a \
        `Time64(Nanosecond)` with microsecond precision, like Spark's `TIME(6)`. \
        Returns NULL if any argument is NULL. \
        Out-of-range fields raise a `DATETIME_FIELD_OUT_OF_BOUNDS` error whether or not ANSI mode is enabled. \
        Seconds that are negative or above 2147483647 are checked first, then hour, minute and second, \
        so the error names the same field as Spark's. \
        Arguments are cast to INT, INT and DECIMAL(16, 6) with DataFusion's cast rather than Spark's, \
        so a value the cast cannot represent (an integer outside the INT range, a malformed string, \
        seconds that do not fit DECIMAL(16, 6), NaN or Infinity) raises a cast error in both ANSI modes. \
        With ANSI mode off Spark wraps the integer or returns NULL instead, and it returns NULL for NaN \
        and Infinity in both modes (https://github.com/apache/datafusion-spark/issues/88).",
    syntax_example = "make_time(hour, minute, second)",
    sql_example = r#"```sql
> SELECT make_time(6, 30, 45.887);
06:30:45.887
> SELECT make_time(23, 59, 59.999999);
23:59:59.999999
> SELECT make_time(1, 18, NULL);
NULL
> SELECT make_time(24, 18, 19.87);
-- error: Invalid value for HourOfDay (valid values 0 - 23): 24
```"#,
    argument(
        name = "hour",
        description = "Hour of the day, 0 to 23. Numeric and string values are cast to INT, \
            and a fractional number is truncated."
    ),
    argument(
        name = "minute",
        description = "Minute of the hour, 0 to 59. Numeric and string values are cast to INT, \
            and a fractional number is truncated."
    ),
    argument(
        name = "second",
        description = "Second of the minute with an optional fraction, 0 to 59.999999. \
            Numeric and string values are cast to DECIMAL(16, 6), so extra fractional digits are rounded half up \
            (59.9999995 becomes 60 and is out of range)."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMakeTime {
    signature: Signature,
}

impl Default for SparkMakeTime {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMakeTime {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

fn out_of_bounds(
    field: &str,
    range: &str,
    value: impl std::fmt::Display,
) -> DataFusionError {
    exec_datafusion_err!(
        "[DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION] Invalid value for {field} (valid values {range}): {value}.  SQLSTATE: 22023"
    )
}

/// Downcasts an argument that `coerce_types` should already have cast. Callers that
/// skip DataFusion's coercion, such as Comet, get an error rather than a panic.
fn primitive_arg<'a, T: ArrowPrimitiveType>(
    array: &'a ArrayRef,
    name: &str,
) -> Result<&'a PrimitiveArray<T>> {
    match array.as_primitive_opt::<T>() {
        Some(array) => Ok(array),
        None => internal_err!(
            "make_time expects {name} as {}, got {}",
            T::DATA_TYPE,
            array.data_type()
        ),
    }
}

/// `secs_micros` is the unscaled value of the `DECIMAL(16, 6)` seconds argument.
fn make_time(hours: i32, minutes: i32, secs_micros: i128) -> Result<i64> {
    let full_secs = secs_micros.div_euclid(MICROS_PER_SECOND);
    let frac_micros = secs_micros.rem_euclid(MICROS_PER_SECOND) as i64;

    // Like Spark's `DateTimeUtils.makeTime`, seconds outside `0..=i32::MAX`
    // are rejected first, so the message reports the whole seconds
    // (truncated toward zero) rather than a wrapped value. Otherwise
    // `LocalTime.of` checks hour, then minute, then second.
    if full_secs < 0 || full_secs > i128::from(i32::MAX) {
        return Err(out_of_bounds(
            "SecondOfMinute",
            "0 - 59",
            secs_micros / MICROS_PER_SECOND,
        ));
    }
    if !(0..=23).contains(&hours) {
        return Err(out_of_bounds("HourOfDay", "0 - 23", hours));
    }
    if !(0..=59).contains(&minutes) {
        return Err(out_of_bounds("MinuteOfHour", "0 - 59", minutes));
    }
    if full_secs > 59 {
        return Err(out_of_bounds("SecondOfMinute", "0 - 59", full_secs));
    }

    // Whole seconds of the day plus the fraction, in micros, then to nanos.
    let day_secs = i64::from(hours) * 3_600 + i64::from(minutes) * 60 + full_secs as i64;
    Ok((day_secs * 1_000_000 + frac_micros) * NANOS_PER_MICRO)
}

impl ScalarUDFImpl for SparkMakeTime {
    fn name(&self) -> &str {
        "make_time"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Time64(TimeUnit::Nanosecond))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [hours, minutes, secs] = take_function_args(self.name(), arg_types)?;
        // Spark's ImplicitCastInputTypes also casts strings to INT / DECIMAL.
        let castable = |t: &DataType| {
            t.is_numeric()
                || *t == DataType::Null
                || matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
        };
        for (arg, t) in [("hour", hours), ("minute", minutes), ("second", secs)] {
            if !castable(t) {
                return exec_err!("make_time expects a numeric {arg}, got {t}");
            }
        }
        Ok(vec![DataType::Int32, DataType::Int32, SECONDS_TYPE])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays: Vec<ArrayRef> = ColumnarValue::values_to_arrays(&args.args)?;
        let [hours, minutes, secs] = take_function_args(self.name(), arrays)?;
        let hours = primitive_arg::<Int32Type>(&hours, "hour")?;
        let minutes = primitive_arg::<Int32Type>(&minutes, "minute")?;
        // The kernel reads the unscaled value as microseconds, so only the scale matters.
        if !matches!(secs.data_type(), DataType::Decimal128(_, 6)) {
            return internal_err!(
                "make_time expects second as Decimal128(_, 6), got {}",
                secs.data_type()
            );
        }
        let secs = primitive_arg::<Decimal128Type>(&secs, "second")?;

        let times: Time64NanosecondArray = (0..hours.len())
            .map(|i| {
                if hours.is_null(i) || minutes.is_null(i) || secs.is_null(i) {
                    Ok(None)
                } else {
                    make_time(hours.value(i), minutes.value(i), secs.value(i)).map(Some)
                }
            })
            .collect::<Result<_>>()?;
        Ok(ColumnarValue::Array(Arc::new(times)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Decimal128Array, Int32Array, Int64Array};
    use arrow::datatypes::{Field, Time64NanosecondType};
    use datafusion_common::config::ConfigOptions;

    /// Invokes the kernel directly, skipping `coerce_types`, as Comet does.
    fn invoke(args: Vec<ArrayRef>) -> Result<ColumnarValue> {
        let arg_fields = args
            .iter()
            .map(|a| Arc::new(Field::new("a", a.data_type().clone(), true)))
            .collect();
        SparkMakeTime::new().invoke_with_args(ScalarFunctionArgs {
            number_rows: args[0].len(),
            args: args.into_iter().map(ColumnarValue::Array).collect(),
            arg_fields,
            return_field: Arc::new(Field::new(
                "r",
                DataType::Time64(TimeUnit::Nanosecond),
                true,
            )),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn int32(v: i32) -> ArrayRef {
        Arc::new(Int32Array::from(vec![v]))
    }

    fn secs(unscaled: i128, precision: u8, scale: i8) -> ArrayRef {
        Arc::new(
            Decimal128Array::from(vec![unscaled])
                .with_precision_and_scale(precision, scale)
                .unwrap(),
        )
    }

    fn assert_err(args: Vec<ArrayRef>, expected: &str) {
        let err = invoke(args).unwrap_err().to_string();
        assert!(err.contains(expected), "{err}");
    }

    #[test]
    fn test_uncoerced_arguments_error() {
        assert_err(
            vec![
                Arc::new(Int64Array::from(vec![1])),
                int32(2),
                secs(3, 16, 6),
            ],
            "make_time expects hour as Int32, got Int64",
        );
        assert_err(
            vec![int32(1), int32(2), int32(3)],
            "make_time expects second as Decimal128(_, 6), got Int32",
        );
        assert_err(
            vec![int32(1), int32(2), secs(3, 10, 2)],
            "make_time expects second as Decimal128(_, 6), got Decimal128(10, 2)",
        );
    }

    #[test]
    fn test_seconds_with_other_precision() -> Result<()> {
        // Only the scale matters, so DECIMAL(38, 6) is read the same way.
        let ColumnarValue::Array(array) =
            invoke(vec![int32(1), int32(2), secs(3_500_000, 38, 6)])?
        else {
            unreachable!("make_time returns an array")
        };
        let array = array.as_primitive::<Time64NanosecondType>();
        assert_eq!(
            array.value(0),
            (3_600 + 2 * 60 + 3) * 1_000_000_000 + 500_000_000
        );
        Ok(())
    }

    fn invoke_array(args: Vec<ArrayRef>) -> Result<Time64NanosecondArray> {
        let ColumnarValue::Array(array) = invoke(args)? else {
            unreachable!("make_time returns an array")
        };
        Ok(array.as_primitive::<Time64NanosecondType>().clone())
    }

    fn nanos(hours: i64, minutes: i64, secs: i64) -> i64 {
        (hours * 3_600 + minutes * 60 + secs) * 1_000_000_000
    }

    #[test]
    fn test_sliced_arrays() -> Result<()> {
        // The invalid fields sit outside the visible slices, so the kernel must
        // honour each array's offset to succeed. The slices also start at
        // different offsets, and their null bitmaps are sliced too.
        let hours: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(99),
            Some(1),
            None,
            Some(3),
            Some(4),
            Some(99),
        ]));
        let minutes: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(99),
            Some(99),
            Some(10),
            Some(20),
            None,
            Some(40),
            Some(99),
        ]));
        let secs: ArrayRef = Arc::new(
            Decimal128Array::from(vec![
                Some(-1),
                Some(-1),
                Some(-1),
                Some(5_000_000),
                Some(6_000_000),
                Some(7_000_000),
                None,
                Some(-1),
            ])
            .with_precision_and_scale(16, 6)
            .unwrap(),
        );

        let result = invoke_array(vec![
            hours.slice(1, 4),
            minutes.slice(2, 4),
            secs.slice(3, 4),
        ])?;
        assert_eq!(
            result,
            Time64NanosecondArray::from(vec![Some(nanos(1, 10, 5)), None, None, None,])
        );
        Ok(())
    }

    #[test]
    fn test_empty_arrays() -> Result<()> {
        let empty_secs: ArrayRef = Arc::new(
            Decimal128Array::from(Vec::<i128>::new())
                .with_precision_and_scale(16, 6)
                .unwrap(),
        );
        let result = invoke_array(vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            empty_secs,
        ])?;
        assert!(result.is_empty());

        // An empty slice of arrays that hold invalid fields.
        let result = invoke_array(vec![
            int32(99).slice(1, 0),
            int32(99).slice(0, 0),
            secs(-1, 16, 6).slice(1, 0),
        ])?;
        assert!(result.is_empty());
        Ok(())
    }
}
