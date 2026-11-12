// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Time64NanosecondArray};
use arrow::datatypes::{DataType, Decimal128Type, Int32Type, TimeUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};

const MICROS_PER_SECOND: i128 = 1_000_000;
const NANOS_PER_MICRO: i64 = 1_000;
const NANOS_PER_SECOND: i64 = 1_000_000_000;

/// Spark's seconds argument type, `DecimalType(16, 6)`.
const SECONDS_PRECISION: u8 = 16;
const SECONDS_SCALE: i8 = 6;

/// Spark-compatible `make_time(hour, minute, second)` (Spark 4.1+).
/// <https://spark.apache.org/docs/latest/api/sql/index.html#make_time>
///
/// Creates a `TIME(6)` from hour (0-23), minute (0-59) and second
/// (0-59.999999) fields, as nanoseconds since midnight (`Time64(Nanosecond)`).
/// The result is NULL if any argument is NULL; out-of-range fields raise
/// `DATETIME_FIELD_OUT_OF_BOUNDS`, regardless of ANSI mode.
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

fn is_string_type(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

fn out_of_bounds(field: &str, range: &str, value: impl std::fmt::Display) -> String {
    format!(
        "[DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION] Invalid value for {field} (valid values {range}): {value}"
    )
}

/// `secs_micros` is the unscaled value of the `DECIMAL(16, 6)` seconds argument.
fn make_time(hours: i32, minutes: i32, secs_micros: i128) -> Result<i64> {
    let full_secs = secs_micros.div_euclid(MICROS_PER_SECOND);
    let frac_micros = secs_micros.rem_euclid(MICROS_PER_SECOND) as i64;

    // Checked first, like Spark, so the message reports the whole seconds
    // (truncated toward zero) rather than a wrapped value.
    if !(0..=59).contains(&full_secs) {
        return exec_err!(
            "{}",
            out_of_bounds("SecondOfMinute", "0 - 59", secs_micros / MICROS_PER_SECOND)
        );
    }
    if !(0..=23).contains(&hours) {
        return exec_err!("{}", out_of_bounds("HourOfDay", "0 - 23", hours));
    }
    if !(0..=59).contains(&minutes) {
        return exec_err!("{}", out_of_bounds("MinuteOfHour", "0 - 59", minutes));
    }

    Ok(i64::from(hours) * 3_600 * NANOS_PER_SECOND
        + i64::from(minutes) * 60 * NANOS_PER_SECOND
        + full_secs as i64 * NANOS_PER_SECOND
        + frac_micros * NANOS_PER_MICRO)
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

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [hours, minutes, secs] = take_function_args(self.name(), arg_types)?;
        // Spark's ImplicitCastInputTypes also casts strings to INT / DECIMAL.
        let castable =
            |t: &DataType| t.is_numeric() || *t == DataType::Null || is_string_type(t);
        for (arg, t) in [("hour", hours), ("minute", minutes), ("second", secs)] {
            if !castable(t) {
                return exec_err!("make_time expects a numeric {arg}, got {t}");
            }
        }
        Ok(vec![
            DataType::Int32,
            DataType::Int32,
            DataType::Decimal128(SECONDS_PRECISION, SECONDS_SCALE),
        ])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let arrays: Vec<ArrayRef> = ColumnarValue::values_to_arrays(&args.args)?;
        let [hours, minutes, secs] = take_function_args(self.name(), arrays)?;
        let hours = hours.as_primitive::<Int32Type>();
        let minutes = minutes.as_primitive::<Int32Type>();
        let secs = secs.as_primitive::<Decimal128Type>();

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

    const HOUR: i64 = 3_600_000_000_000;
    const MINUTE: i64 = 60_000_000_000;

    #[test]
    fn valid_times() {
        assert_eq!(make_time(0, 0, 0).unwrap(), 0);
        assert_eq!(make_time(1, 0, 0).unwrap(), HOUR);
        assert_eq!(make_time(0, 1, 0).unwrap(), MINUTE);
        assert_eq!(make_time(0, 0, 1_500_000).unwrap(), 1_500_000_000);
        assert_eq!(make_time(23, 59, 59_999_999).unwrap(), 86_399_999_999_000);
        // make_time(1, 18, 19.87) = 01:18:19.87
        assert_eq!(
            make_time(1, 18, 19_870_000).unwrap(),
            HOUR + 18 * MINUTE + 19_870_000_000
        );
    }

    #[test]
    fn out_of_range_fields_report_the_spark_message() {
        let msg = |r: Result<i64>| r.unwrap_err().to_string();
        assert!(
            msg(make_time(-1, 18, 19_870_000))
                .contains("Invalid value for HourOfDay (valid values 0 - 23): -1")
        );
        assert!(msg(make_time(24, 0, 0)).contains("HourOfDay"));
        assert!(
            msg(make_time(1, 60, 19_870_000))
                .contains("Invalid value for MinuteOfHour (valid values 0 - 59): 60")
        );
        assert!(
            msg(make_time(1, 18, 60_000_000))
                .contains("Invalid value for SecondOfMinute (valid values 0 - 59): 60")
        );
        // Whole seconds are truncated toward zero in the message, like Spark.
        assert!(msg(make_time(1, 18, -999_999_999_999_999)).contains(": -999999999"));
        assert!(msg(make_time(1, 18, 4_294_967_297_999_999)).contains(": 4294967297"));
    }

    #[test]
    fn coerce_types_rules() -> Result<()> {
        let f = SparkMakeTime::new();
        let dec = DataType::Decimal128(SECONDS_PRECISION, SECONDS_SCALE);
        let got =
            f.coerce_types(&[DataType::Int64, DataType::Int8, DataType::Float64])?;
        assert_eq!(got, vec![DataType::Int32, DataType::Int32, dec]);
        // strings are cast like Spark; other types are rejected
        let got = f.coerce_types(&[DataType::Utf8, DataType::Int32, DataType::Utf8])?;
        assert_eq!(got[0], DataType::Int32);
        assert!(
            f.coerce_types(&[DataType::Boolean, DataType::Int32, DataType::Int32])
                .is_err()
        );
        Ok(())
    }
}
