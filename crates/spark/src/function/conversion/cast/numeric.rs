// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Casts from numeric types.
//!
//! Ported from Comet's `native/spark-expr/src/conversion_funcs/numeric.rs`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ArrowPrimitiveType, AsArray, PrimitiveArray};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType,
};
use arrow::temporal_conversions::MICROSECONDS;
use datafusion_common::{DataFusionError, Result, internal_err};

use crate::{EvalMode, SparkError};

/// Casts integral seconds since the epoch to a timestamp, like Spark's
/// `SECONDS.toMicros`, which saturates on overflow in every eval mode.
pub(super) fn cast_int_to_timestamp(
    array: &ArrayRef,
    target_tz: &Option<Arc<str>>,
) -> Result<ArrayRef> {
    fn micros<T>(array: &ArrayRef) -> PrimitiveArray<TimestampMicrosecondType>
    where
        T: ArrowPrimitiveType,
        T::Native: Into<i64>,
    {
        array
            .as_primitive::<T>()
            .unary(|seconds| seconds.into().saturating_mul(MICROSECONDS))
    }

    let micros = match array.data_type() {
        DataType::Int8 => micros::<Int8Type>(array),
        DataType::Int16 => micros::<Int16Type>(array),
        DataType::Int32 => micros::<Int32Type>(array),
        DataType::Int64 => micros::<Int64Type>(array),
        other => return internal_err!("cast_int_to_timestamp: unexpected {other}"),
    };
    Ok(Arc::new(micros.with_timezone_opt(target_tz.clone())))
}

/// Casts fractional seconds since the epoch to a timestamp. Spark widens a
/// `FLOAT` to `DOUBLE` first.
///
/// | Mode   | NaN or infinite      | Out of range                   |
/// |--------|----------------------|--------------------------------|
/// | LEGACY | NULL                 | Saturates, like Scala's `toLong` |
/// | ANSI   | `CAST_INVALID_INPUT` | `CAST_OVERFLOW`                |
/// | TRY    | NULL                 | NULL                           |
///
/// Comet returns NULL for an out-of-range value in LEGACY mode, while Spark's
/// `doubleToTimestamp` saturates.
pub(super) fn cast_float_to_timestamp(
    array: &ArrayRef,
    target_tz: &Option<Arc<str>>,
    eval_mode: EvalMode,
) -> Result<ArrayRef> {
    let micros = match array.data_type() {
        DataType::Float32 => {
            float_to_timestamp(array.as_primitive::<Float32Type>(), eval_mode)?
        }
        DataType::Float64 => {
            float_to_timestamp(array.as_primitive::<Float64Type>(), eval_mode)?
        }
        other => return internal_err!("cast_float_to_timestamp: unexpected {other}"),
    };
    Ok(Arc::new(micros.with_timezone_opt(target_tz.clone())))
}

fn float_to_timestamp<T>(
    array: &PrimitiveArray<T>,
    eval_mode: EvalMode,
) -> Result<PrimitiveArray<TimestampMicrosecondType>>
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64>,
{
    Ok(match eval_mode {
        EvalMode::Legacy => float_to_micros_nulling(array, f64::is_finite),
        EvalMode::Try => float_to_micros_nulling(array, |seconds| {
            float_timestamp_micros(seconds).is_some()
        }),
        // `try_unary` skips null slots, so a NaN under a null is not raised.
        EvalMode::Ansi => array.try_unary(|value| {
            let seconds = value.into();
            float_timestamp_micros(seconds)
                .ok_or_else(|| DataFusionError::from(float_timestamp_error(seconds)))
        })?,
    })
}

/// Converts seconds to microseconds, and nulls the rows where `is_valid` is
/// false. `as` saturates, like Scala's `Double.toLong`.
///
/// The conversion and the validity check are separate passes, so both
/// vectorize, and an array where nothing becomes null keeps its null buffer.
fn float_to_micros_nulling<T>(
    array: &PrimitiveArray<T>,
    is_valid: impl Fn(f64) -> bool,
) -> PrimitiveArray<TimestampMicrosecondType>
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64>,
{
    let micros = array.unary(|value| (value.into() * MICROSECONDS as f64) as i64);
    let values = array.values();
    let valid = BooleanBuffer::collect_bool(array.len(), |i| is_valid(values[i].into()));
    if valid.count_set_bits() == array.len() {
        return micros;
    }
    let nulls = NullBuffer::union(array.nulls(), Some(&NullBuffer::new(valid)));
    let (_, values, _) = micros.into_parts();
    PrimitiveArray::new(values, nulls)
}

/// Seconds since the epoch as microseconds, or `None` when the value is NaN,
/// infinite, or out of range, like Spark's `doubleToTimestampAnsi`.
fn float_timestamp_micros(seconds: f64) -> Option<i64> {
    // Spark's check from `DoubleExactNumeric.toLong` is
    // `floor(x) <= Long.MaxValue && ceil(x) >= Long.MinValue`, with the bounds
    // converted to double. Doubles that large are whole numbers, so it is a
    // bounds check, which NaN and infinity fail. A value that passes is
    // truncated toward zero.
    let micros = seconds * MICROSECONDS as f64;
    (i64::MIN as f64..=i64::MAX as f64)
        .contains(&micros)
        .then_some(micros as i64)
}

/// The ANSI error for a value [`float_timestamp_micros`] rejects. The source
/// type is `DOUBLE` even for a `FLOAT` array, because Spark widens it first.
fn float_timestamp_error(seconds: f64) -> SparkError {
    if !seconds.is_finite() {
        SparkError::InvalidInputInCastToDatetime {
            value: java_double_literal(seconds),
            from_type: "DOUBLE".to_string(),
            to_type: "TIMESTAMP".to_string(),
        }
    } else {
        // The overflow is in the conversion of the product to a long.
        SparkError::CastOverflow {
            value: java_double_literal(seconds * MICROSECONDS as f64),
            from_type: "DOUBLE".to_string(),
            to_type: "BIGINT".to_string(),
        }
    }
}

/// A double that is NaN, infinite, or out of the range of a long, as Spark's
/// `toSQLValue` writes it, such as `NaN` or `1.0E25D`. Java's `Double.toString`
/// uses scientific notation from 10^7, so every value out of the range of a
/// long is in it.
fn java_double_literal(value: f64) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    if mantissa.contains('.') {
        format!("{mantissa}E{exponent}D")
    } else {
        format!("{mantissa}.0E{exponent}D")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Float32Array, Float64Array, Int8Array, Int64Array, TimestampMicrosecondArray,
    };

    fn utc() -> Option<Arc<str>> {
        Some("UTC".into())
    }

    fn micros(array: &ArrayRef) -> &TimestampMicrosecondArray {
        array.as_primitive::<TimestampMicrosecondType>()
    }

    #[test]
    fn test_int_to_timestamp() {
        let array: ArrayRef = Arc::new(Int8Array::from(vec![Some(-128), None, Some(1)]));
        let result = cast_int_to_timestamp(&array, &utc()).unwrap();
        assert_eq!(
            micros(&result),
            &TimestampMicrosecondArray::from(vec![
                Some(-128_000_000),
                None,
                Some(1_000_000)
            ])
            .with_timezone("UTC")
        );
    }

    #[test]
    fn test_int_to_timestamp_saturates() {
        let array: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX, i64::MIN]));
        let result = cast_int_to_timestamp(&array, &None).unwrap();
        assert_eq!(micros(&result).values(), &[i64::MAX, i64::MIN]);
        assert_eq!(
            result.data_type(),
            &DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
        );
    }

    fn float_input() -> ArrayRef {
        Arc::new(Float64Array::from(vec![
            Some(1.5),
            Some(-1.0000005),
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(1e19),
            Some(-1e19),
        ]))
    }

    #[test]
    fn test_float_to_timestamp_legacy() {
        let result =
            cast_float_to_timestamp(&float_input(), &utc(), EvalMode::Legacy).unwrap();
        assert_eq!(
            micros(&result),
            &TimestampMicrosecondArray::from(vec![
                Some(1_500_000),
                // Truncated toward zero.
                Some(-1_000_000),
                None,
                None,
                None,
                Some(i64::MAX),
                Some(i64::MIN),
            ])
            .with_timezone("UTC")
        );
    }

    #[test]
    fn test_float_to_timestamp_try() {
        let result =
            cast_float_to_timestamp(&float_input(), &utc(), EvalMode::Try).unwrap();
        assert_eq!(
            micros(&result),
            &TimestampMicrosecondArray::from(vec![
                Some(1_500_000),
                Some(-1_000_000),
                None,
                None,
                None,
                None,
                None,
            ])
            .with_timezone("UTC")
        );
    }

    #[test]
    fn test_float_to_timestamp_ansi() {
        let cast = |value: f64| {
            let array: ArrayRef = Arc::new(Float64Array::from(vec![value]));
            cast_float_to_timestamp(&array, &utc(), EvalMode::Ansi)
        };
        assert_eq!(micros(&cast(1.5).unwrap()).value(0), 1_500_000);

        let error = cast(f64::NAN).unwrap_err();
        assert_eq!(
            SparkError::find(&error),
            Some(&SparkError::InvalidInputInCastToDatetime {
                value: "NaN".to_string(),
                from_type: "DOUBLE".to_string(),
                to_type: "TIMESTAMP".to_string(),
            }),
            "{error}"
        );
        let error = cast(f64::NEG_INFINITY).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("The value -Infinity of the type \"DOUBLE\""),
            "{error}"
        );

        let error = cast(1e19).unwrap_err();
        assert_eq!(
            SparkError::find(&error),
            Some(&SparkError::CastOverflow {
                value: "1.0E25D".to_string(),
                from_type: "DOUBLE".to_string(),
                to_type: "BIGINT".to_string(),
            }),
            "{error}"
        );

        // A NaN under a null is not an error.
        let array: ArrayRef = Arc::new(Float64Array::new(
            vec![f64::NAN].into(),
            Some(vec![false].into()),
        ));
        let result = cast_float_to_timestamp(&array, &utc(), EvalMode::Ansi).unwrap();
        assert!(result.is_null(0));
    }

    #[test]
    fn test_float_to_timestamp_keeps_null_buffer_when_nothing_becomes_null() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0]));
        for eval_mode in [EvalMode::Legacy, EvalMode::Try] {
            let result = cast_float_to_timestamp(&array, &None, eval_mode).unwrap();
            assert!(result.nulls().is_none(), "{eval_mode:?}");
        }
    }

    #[test]
    fn test_float32_widens_to_double() {
        let array: ArrayRef = Arc::new(Float32Array::from(vec![0.1f32]));
        let result = cast_float_to_timestamp(&array, &None, EvalMode::Legacy).unwrap();
        // 0.1f32 is 0.100000001490116... as a double.
        assert_eq!(micros(&result).value(0), 100_000);
    }

    #[test]
    fn test_java_double_literal() {
        assert_eq!(java_double_literal(1e25), "1.0E25D");
        assert_eq!(java_double_literal(-1.2345e19), "-1.2345E19D");
        assert_eq!(java_double_literal(f64::INFINITY), "Infinity");
    }
}
