// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Time64NanosecondArray};
use arrow::datatypes::{DataType, Decimal128Type, Int32Type, TimeUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{DataFusionError, Result, exec_datafusion_err, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;

const MICROS_PER_SECOND: i128 = 1_000_000;
const NANOS_PER_MICRO: i64 = 1_000;

/// Spark's seconds argument type, `DecimalType(16, 6)`.
const SECONDS_PRECISION: u8 = 16;
const SECONDS_SCALE: i8 = 6;

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
        so the error names the same field as Spark's.",
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
        description = "Hour of the day, 0 to 23. Numeric and string values are cast to INT."
    ),
    argument(
        name = "minute",
        description = "Minute of the hour, 0 to 59. Numeric and string values are cast to INT."
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
        "[DATETIME_FIELD_OUT_OF_BOUNDS.WITHOUT_SUGGESTION] Invalid value for {field} (valid values {range}): {value}"
    )
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
