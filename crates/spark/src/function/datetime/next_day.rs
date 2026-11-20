// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use crate::function::nullability::{SparkNullability, spark_nullability};
use arrow::array::{
    Array, ArrayRef, AsArray, Date32Array, Date32Builder, StringArrayType,
};
use arrow::datatypes::{DataType, Date32Type, FieldRef};
use chrono::{Datelike, Weekday};
use datafusion_common::types::{NativeType, logical_date, logical_string};
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark's `NextDay` is null-intolerant and always nullable, because an unknown day of the
/// week returns NULL when ANSI mode is off.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant_always_nullable();

/// Spark-compatible `next_day` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#next_day>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the first date after `start_date` that falls on `day_of_week`. \
        The result is always later than `start_date`: if `start_date` is already that day, \
        the date one week later is returned. DataFusion has no built-in `next_day`. \
        Strings and timestamps are cast to DATE first. \
        Returns NULL if either input is NULL, or if `day_of_week` is not a recognized day name \
        (including names padded with whitespace) when ANSI mode is disabled. Spark raises an \
        `[ILLEGAL_DAY_OF_WEEK]` error for an unrecognized day name when ANSI mode is enabled.",
    syntax_example = "next_day(start_date, day_of_week)",
    sql_example = r#"```sql
> SELECT next_day(DATE '2015-01-14', 'TU');
2015-01-20
> SELECT next_day(DATE '2015-07-27', 'Sun');
2015-08-02
> SELECT next_day(TIMESTAMP '2015-07-23 12:12:12', 'Mon');
2015-07-27
> SELECT next_day(DATE '2020-01-01', 'invalid_day');
NULL
```"#,
    argument(
        name = "start_date",
        description = "DATE, TIMESTAMP or string expression. NULL returns NULL."
    ),
    argument(
        name = "day_of_week",
        description = "Day of the week, case-insensitive: a two-letter abbreviation (`MO`), a three-letter abbreviation (`Mon`) or the full name (`Monday`). Other types are cast to string. NULL or an unrecognized value returns NULL (or an error when `datafusion.execution.enable_ansi_mode` is true)."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkNextDay {
    signature: Signature,
}

impl Default for SparkNextDay {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkNextDay {
    pub fn new() -> Self {
        Self {
            // Spark casts the start date to DATE, so STRING and TIMESTAMP are accepted
            signature: Signature::coercible(
                vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_date()),
                        vec![
                            TypeSignatureClass::Native(logical_string()),
                            TypeSignatureClass::Timestamp,
                        ],
                        NativeType::Date,
                    ),
                    Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_string()),
                        // Preserve the previous Exact signature's weekday-to-string coercion.
                        vec![TypeSignatureClass::Any],
                        NativeType::String,
                    ),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkNextDay {
    fn name(&self) -> &str {
        "next_day"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(NULLABILITY.return_field(self.name(), DataType::Date32, args.arg_fields))
    }

    spark_nullability!(NULLABILITY);

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        let [date, day_of_week] = args.args.as_slice() else {
            return exec_err!(
                "Spark `next_day` function requires 2 arguments, got {}",
                args.args.len()
            );
        };

        match (date, day_of_week) {
            (ColumnarValue::Scalar(date), ColumnarValue::Scalar(day_of_week)) => {
                match (date, day_of_week) {
                    (
                        ScalarValue::Date32(days),
                        ScalarValue::Utf8(day_of_week)
                        | ScalarValue::LargeUtf8(day_of_week)
                        | ScalarValue::Utf8View(day_of_week),
                    ) => {
                        if let (Some(days), Some(day_of_week)) = (days, day_of_week) {
                            Ok(ColumnarValue::Scalar(ScalarValue::Date32(
                                spark_next_day(
                                    *days,
                                    day_of_week.as_str(),
                                    enable_ansi_mode,
                                )?,
                            )))
                        } else {
                            Ok(ColumnarValue::Scalar(ScalarValue::Date32(None)))
                        }
                    }
                    _ => exec_err!(
                        "Spark `next_day` function: first arg must be date, second arg must be string. Got {:?}",
                        args.args
                    ),
                }
            }
            (ColumnarValue::Array(date_array), ColumnarValue::Scalar(day_of_week)) => {
                match (date_array.data_type(), day_of_week) {
                    (
                        DataType::Date32,
                        ScalarValue::Utf8(day_of_week)
                        | ScalarValue::LargeUtf8(day_of_week)
                        | ScalarValue::Utf8View(day_of_week),
                    ) => {
                        let date_array = date_array.as_primitive::<Date32Type>();
                        if let Some(day_of_week) = day_of_week {
                            match parse_day_of_week(day_of_week.as_str()) {
                                Some(target_dow) => {
                                    let result: Date32Array = date_array
                                        .unary_opt(|days| {
                                            compute_next_day(days, target_dow)
                                        })
                                        .with_data_type(DataType::Date32);
                                    Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
                                }
                                None => {
                                    if enable_ansi_mode
                                        && (date_array.null_count() < date_array.len())
                                    {
                                        exec_err!(
                                            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: {day_of_week}. SQLSTATE: 22009"
                                        )
                                    } else {
                                        let result =
                                            Date32Array::new_null(date_array.len());
                                        Ok(ColumnarValue::Array(
                                            Arc::new(result) as ArrayRef
                                        ))
                                    }
                                }
                            }
                        } else {
                            let result = Date32Array::new_null(date_array.len());
                            Ok(ColumnarValue::Array(Arc::new(result) as ArrayRef))
                        }
                    }
                    _ => exec_err!(
                        "Spark `next_day` function: first arg must be date, second arg must be string. Got {:?}",
                        args.args
                    ),
                }
            }
            (ColumnarValue::Scalar(date), ColumnarValue::Array(day_of_week_array)) => {
                match (date, day_of_week_array.data_type()) {
                    (
                        ScalarValue::Date32(days),
                        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
                    ) => {
                        let date_array = match days {
                            Some(d) => {
                                Date32Array::from_value(*d, day_of_week_array.len())
                            }
                            None => Date32Array::new_null(day_of_week_array.len()),
                        };
                        let result = match day_of_week_array.data_type() {
                            DataType::Utf8 => {
                                let day_of_week_array =
                                    day_of_week_array.as_string::<i32>();
                                process_next_day_arrays(
                                    &date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            DataType::LargeUtf8 => {
                                let day_of_week_array =
                                    day_of_week_array.as_string::<i64>();
                                process_next_day_arrays(
                                    &date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            DataType::Utf8View => {
                                let day_of_week_array =
                                    day_of_week_array.as_string_view();
                                process_next_day_arrays(
                                    &date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            _ => unreachable!(),
                        }?;
                        Ok(ColumnarValue::Array(result))
                    }
                    _ => exec_err!(
                        "Spark `next_day` function: first arg must be date, second arg must be string. Got {:?}",
                        args.args
                    ),
                }
            }
            (
                ColumnarValue::Array(date_array),
                ColumnarValue::Array(day_of_week_array),
            ) => {
                let result = match (date_array.data_type(), day_of_week_array.data_type())
                {
                    (
                        DataType::Date32,
                        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View,
                    ) => {
                        let date_array: &Date32Array =
                            date_array.as_primitive::<Date32Type>();
                        match day_of_week_array.data_type() {
                            DataType::Utf8 => {
                                let day_of_week_array =
                                    day_of_week_array.as_string::<i32>();
                                process_next_day_arrays(
                                    date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            DataType::LargeUtf8 => {
                                let day_of_week_array =
                                    day_of_week_array.as_string::<i64>();
                                process_next_day_arrays(
                                    date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            DataType::Utf8View => {
                                let day_of_week_array =
                                    day_of_week_array.as_string_view();
                                process_next_day_arrays(
                                    date_array,
                                    day_of_week_array,
                                    enable_ansi_mode,
                                )
                            }
                            other => {
                                exec_err!(
                                    "Spark `next_day` function: second arg must be string. Got {other:?}"
                                )
                            }
                        }
                    }
                    (left, right) => {
                        exec_err!(
                            "Spark `next_day` function: first arg must be date, second arg must be string. Got {left:?}, {right:?}"
                        )
                    }
                }?;
                Ok(ColumnarValue::Array(result))
            }
        }
    }
}

fn process_next_day_arrays<'a, S>(
    date_array: &Date32Array,
    day_of_week_array: &'a S,
    enable_ansi_mode: bool,
) -> Result<ArrayRef>
where
    &'a S: StringArrayType<'a>,
{
    let mut builder = Date32Builder::with_capacity(date_array.len());
    for (days, day_of_week) in date_array.iter().zip(day_of_week_array.iter()) {
        match (days, day_of_week) {
            (Some(days), Some(day_of_week)) => match parse_day_of_week(day_of_week) {
                Some(target_dow) => {
                    builder.append_option(compute_next_day(days, target_dow));
                }
                None => {
                    if enable_ansi_mode {
                        return exec_err!(
                            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: {day_of_week}. SQLSTATE: 22009"
                        );
                    } else {
                        builder.append_null();
                    }
                }
            },
            _ => builder.append_null(),
        }
    }
    Ok(Arc::new(builder.finish()) as ArrayRef)
}

fn parse_day_of_week(day_of_week: &str) -> Option<Weekday> {
    match day_of_week.to_uppercase().as_str() {
        "MO" | "MON" | "MONDAY" => Some(Weekday::Mon),
        "TU" | "TUE" | "TUESDAY" => Some(Weekday::Tue),
        "WE" | "WED" | "WEDNESDAY" => Some(Weekday::Wed),
        "TH" | "THU" | "THURSDAY" => Some(Weekday::Thu),
        "FR" | "FRI" | "FRIDAY" => Some(Weekday::Fri),
        "SA" | "SAT" | "SATURDAY" => Some(Weekday::Sat),
        "SU" | "SUN" | "SUNDAY" => Some(Weekday::Sun),
        _ => None,
    }
}

fn compute_next_day(days: i32, target_dow: Weekday) -> Option<i32> {
    let date = Date32Type::to_naive_date_opt(days)?;
    let delta = 7 - date.weekday().days_since(target_dow) as i32;
    days.checked_add(delta)
}

fn spark_next_day(
    days: i32,
    day_of_week: &str,
    enable_ansi_mode: bool,
) -> Result<Option<i32>> {
    match parse_day_of_week(day_of_week) {
        Some(target_dow) => Ok(compute_next_day(days, target_dow)),
        None => {
            if enable_ansi_mode {
                exec_err!(
                    "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: {day_of_week}. SQLSTATE: 22009"
                )
            } else {
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;

    #[test]
    fn return_type_is_not_used() {
        let func = SparkNextDay::new();
        let err = func
            .return_type(&[DataType::Date32, DataType::Utf8])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("return_field_from_args should be used instead")
        );
    }

    #[test]
    fn next_day_is_always_nullable() {
        let func = SparkNextDay::new();
        let date_field: FieldRef =
            Arc::new(Field::new("start_date", DataType::Date32, false));
        let day_field: FieldRef =
            Arc::new(Field::new("day_of_week", DataType::Utf8, false));

        let field = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[Arc::clone(&date_field), Arc::clone(&day_field)],
                scalar_arguments: &[None, None],
            })
            .unwrap();

        assert_eq!(field.data_type(), &DataType::Date32);
        assert!(field.is_nullable());
    }

    #[test]
    fn next_day_rejects_whitespace_padded_day_names() {
        let monday = 19723; // 2024-01-01
        assert_eq!(spark_next_day(monday, " MO ", false).unwrap(), None);

        let err = spark_next_day(monday, " MO ", true).unwrap_err();
        assert!(
            err.to_string()
                .contains("[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week:  MO . SQLSTATE: 22009"),
            "unexpected error message: {err}"
        );

        // Leading whitespace
        assert_eq!(spark_next_day(monday, " MO", false).unwrap(), None);
        let err = spark_next_day(monday, " MO", true).unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week:  MO. SQLSTATE: 22009"
        ));

        // Trailing whitespace
        assert_eq!(spark_next_day(monday, "MO ", false).unwrap(), None);
        let err = spark_next_day(monday, "MO ", true).unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: MO . SQLSTATE: 22009"
        ));

        // Empty string
        assert_eq!(spark_next_day(monday, "", false).unwrap(), None);
        let err = spark_next_day(monday, "", true).unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: . SQLSTATE: 22009"
        ));

        // 4-letter invalid abbreviation
        assert_eq!(spark_next_day(monday, "TUES", false).unwrap(), None);
        let err = spark_next_day(monday, "TUES", true).unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: TUES. SQLSTATE: 22009"
        ));

        // Non-ASCII string
        assert_eq!(spark_next_day(monday, "月曜日", false).unwrap(), None);
        let err = spark_next_day(monday, "月曜日", true).unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: 月曜日. SQLSTATE: 22009"
        ));

        // Case-insensitive valid strings
        assert_eq!(spark_next_day(monday, "mo", true).unwrap(), Some(19730));
        assert_eq!(spark_next_day(monday, "mOnDaY", true).unwrap(), Some(19730));
    }

    #[test]
    fn next_day_handles_far_future_start_dates() {
        // Regression for #23891: for start dates near the end of the
        // representable `Date32` range, the next occurrence can land past
        // `chrono::NaiveDate::MAX` (epoch day 95026236). Computing the result
        // on the epoch day directly (as Spark does) must return a value rather
        // than panicking with `NaiveDate + TimeDelta overflowed`.
        //
        // 95026236 is a Monday, so `next_day(.., "Mon")` advances a full week.
        assert_eq!(
            spark_next_day(95026236, "Mon", false).unwrap(),
            Some(95026243)
        );
        assert_eq!(
            spark_next_day(95026230, "Tue", true).unwrap(),
            Some(95026237)
        );
    }

    #[test]
    fn next_day_ansi_mode_scalar_evaluation() {
        let func = SparkNextDay::new();
        let monday = 19723; // 2024-01-01

        let make_args = |date: ScalarValue, dow: ScalarValue, ansi: bool| {
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi;
            ScalarFunctionArgs {
                args: vec![ColumnarValue::Scalar(date), ColumnarValue::Scalar(dow)],
                arg_fields: vec![
                    Arc::new(Field::new("start_date", DataType::Date32, true)),
                    Arc::new(Field::new("day_of_week", DataType::Utf8, true)),
                ],
                number_rows: 1,
                return_field: Arc::new(Field::new("next_day", DataType::Date32, true)),
                config_options: Arc::new(config),
            }
        };

        // Valid day with ANSI mode on: returns next Tuesday (2024-01-02 -> 19724)
        let res = func
            .invoke_with_args(make_args(
                ScalarValue::Date32(Some(monday)),
                ScalarValue::Utf8(Some("TU".to_string())),
                true,
            ))
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::Date32(val)) = res else {
            panic!("expected scalar Date32")
        };
        assert_eq!(val, Some(19724));

        // Invalid day with ANSI mode off: returns NULL
        let res = func
            .invoke_with_args(make_args(
                ScalarValue::Date32(Some(monday)),
                ScalarValue::Utf8(Some("invalid".to_string())),
                false,
            ))
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::Date32(val)) = res else {
            panic!("expected scalar Date32")
        };
        assert_eq!(val, None);

        // Invalid day with ANSI mode on: raises Spark ILLEGAL_DAY_OF_WEEK error
        let err = func
            .invoke_with_args(make_args(
                ScalarValue::Date32(Some(monday)),
                ScalarValue::Utf8(Some("invalid".to_string())),
                true,
            ))
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: invalid. SQLSTATE: 22009"),
            "unexpected error message: {err}"
        );

        // NULL start date with ANSI mode on: short-circuits and returns NULL
        let res = func
            .invoke_with_args(make_args(
                ScalarValue::Date32(None),
                ScalarValue::Utf8(Some("invalid".to_string())),
                true,
            ))
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::Date32(val)) = res else {
            panic!("expected scalar Date32")
        };
        assert_eq!(val, None);

        // NULL day_of_week with ANSI mode on: returns NULL
        let res = func
            .invoke_with_args(make_args(
                ScalarValue::Date32(Some(monday)),
                ScalarValue::Utf8(None),
                true,
            ))
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::Date32(val)) = res else {
            panic!("expected scalar Date32")
        };
        assert_eq!(val, None);
    }

    #[test]
    fn next_day_ansi_mode_array_evaluation() {
        let func = SparkNextDay::new();

        let make_array_args = |dates: Date32Array, dow: ScalarValue, ansi: bool| {
            let num_rows = dates.len();
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi;
            ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(dates) as ArrayRef),
                    ColumnarValue::Scalar(dow),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("start_date", DataType::Date32, true)),
                    Arc::new(Field::new("day_of_week", DataType::Utf8, true)),
                ],
                number_rows: num_rows,
                return_field: Arc::new(Field::new("next_day", DataType::Date32, true)),
                config_options: Arc::new(config),
            }
        };

        // Array with all NULL dates does not trigger error even with invalid day under ANSI
        let all_nulls = Date32Array::new_null(3);
        let res = func
            .invoke_with_args(make_array_args(
                all_nulls,
                ScalarValue::Utf8(Some("invalid".to_string())),
                true,
            ))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 3);

        // Array with non-null date triggers error with invalid day under ANSI
        let mixed = Date32Array::from(vec![None, Some(19723), None]);
        let err = func
            .invoke_with_args(make_array_args(
                mixed,
                ScalarValue::Utf8(Some("xx".to_string())),
                true,
            ))
            .unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: xx. SQLSTATE: 22009"
        ));

        // Array with non-null date under ANSI off: returns all nulls
        let mixed = Date32Array::from(vec![None, Some(19723), None]);
        let res = func
            .invoke_with_args(make_array_args(
                mixed,
                ScalarValue::Utf8(Some("xx".to_string())),
                false,
            ))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.null_count(), 3);
    }

    #[test]
    fn next_day_scalar_date_array_dow_evaluation() {
        use arrow::array::StringArray;

        let func = SparkNextDay::new();
        let monday = 19723; // 2024-01-01

        let make_args = |date: ScalarValue, dows: StringArray, ansi: bool| {
            let num_rows = dows.len();
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi;
            ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Scalar(date),
                    ColumnarValue::Array(Arc::new(dows) as ArrayRef),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("start_date", DataType::Date32, true)),
                    Arc::new(Field::new("day_of_week", DataType::Utf8, true)),
                ],
                number_rows: num_rows,
                return_field: Arc::new(Field::new("next_day", DataType::Date32, true)),
                config_options: Arc::new(config),
            }
        };

        // Scalar date + Array of valid DOWs with ANSI ON: returns calculated dates
        let dows = StringArray::from(vec![Some("TU"), Some("WE")]);
        let res = func
            .invoke_with_args(make_args(ScalarValue::Date32(Some(monday)), dows, true))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert_eq!(date_arr.value(0), 19724); // 2024-01-02
        assert_eq!(date_arr.value(1), 19725); // 2024-01-03

        // Scalar date + Array with NULL DOW: NULL propagates
        let dows = StringArray::from(vec![Some("TU"), None, Some("WE")]);
        let res = func
            .invoke_with_args(make_args(ScalarValue::Date32(Some(monday)), dows, true))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert_eq!(date_arr.value(0), 19724);
        assert!(date_arr.is_null(1));
        assert_eq!(date_arr.value(2), 19725);

        // Scalar date + Array with invalid DOW with ANSI ON: errors
        let dows = StringArray::from(vec![Some("TU"), Some("invalid")]);
        let err = func
            .invoke_with_args(make_args(ScalarValue::Date32(Some(monday)), dows, true))
            .unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: invalid. SQLSTATE: 22009"
        ));

        // Scalar date + Array with invalid DOW with ANSI OFF: returns NULL for invalid row
        let dows = StringArray::from(vec![Some("TU"), Some("invalid")]);
        let res = func
            .invoke_with_args(make_args(ScalarValue::Date32(Some(monday)), dows, false))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert_eq!(date_arr.value(0), 19724);
        assert!(date_arr.is_null(1));

        // Scalar NULL date + Array with invalid DOW with ANSI ON: short-circuits to all NULLs
        let dows = StringArray::from(vec![Some("TU"), Some("invalid")]);
        let res = func
            .invoke_with_args(make_args(ScalarValue::Date32(None), dows, true))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.null_count(), 2);
    }

    #[test]
    fn next_day_array_date_array_dow_evaluation() {
        use arrow::array::StringArray;

        let func = SparkNextDay::new();
        let monday = 19723; // 2024-01-01

        let make_args = |dates: Date32Array, dows: StringArray, ansi: bool| {
            let num_rows = dates.len();
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi;
            ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(dates) as ArrayRef),
                    ColumnarValue::Array(Arc::new(dows) as ArrayRef),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("start_date", DataType::Date32, true)),
                    Arc::new(Field::new("day_of_week", DataType::Utf8, true)),
                ],
                number_rows: num_rows,
                return_field: Arc::new(Field::new("next_day", DataType::Date32, true)),
                config_options: Arc::new(config),
            }
        };

        // Array dates + Array valid DOWs with ANSI ON
        let dates = Date32Array::from(vec![Some(monday), Some(monday)]);
        let dows = StringArray::from(vec![Some("TU"), Some("WE")]);
        let res = func.invoke_with_args(make_args(dates, dows, true)).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert_eq!(date_arr.value(0), 19724);
        assert_eq!(date_arr.value(1), 19725);

        // Array dates + Array with invalid DOW with ANSI ON: errors
        let dates = Date32Array::from(vec![Some(monday), Some(monday)]);
        let dows = StringArray::from(vec![Some("TU"), Some("invalid")]);
        let err = func
            .invoke_with_args(make_args(dates, dows, true))
            .unwrap_err();
        assert!(err.to_string().contains(
            "[ILLEGAL_DAY_OF_WEEK] Illegal input for day of week: invalid. SQLSTATE: 22009"
        ));

        // Array dates + Array with invalid DOW with ANSI OFF: returns NULL for invalid row
        let dates = Date32Array::from(vec![Some(monday), Some(monday)]);
        let dows = StringArray::from(vec![Some("TU"), Some("invalid")]);
        let res = func
            .invoke_with_args(make_args(dates, dows, false))
            .unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert_eq!(date_arr.value(0), 19724);
        assert!(date_arr.is_null(1));

        // Array with NULL date at row 0: short-circuits to NULL even with invalid DOW at row 0
        let dates = Date32Array::from(vec![None, Some(monday)]);
        let dows = StringArray::from(vec![Some("invalid"), Some("TU")]);
        let res = func.invoke_with_args(make_args(dates, dows, true)).unwrap();
        let ColumnarValue::Array(arr) = res else {
            panic!("expected array")
        };
        let date_arr = arr.as_primitive::<Date32Type>();
        assert!(date_arr.is_null(0));
        assert_eq!(date_arr.value(1), 19724);
    }

    #[test]
    fn next_day_unicode_day_of_week_case_folding() {
        let func = SparkNextDay::new();
        let monday = 19723; // 2024-01-01

        let make_args = |date: ScalarValue, dow: &str, ansi: bool| {
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi;
            ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Scalar(date),
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some(dow.to_string()))),
                ],
                arg_fields: vec![
                    Arc::new(Field::new("start_date", DataType::Date32, true)),
                    Arc::new(Field::new("day_of_week", DataType::Utf8, true)),
                ],
                number_rows: 1,
                return_field: Arc::new(Field::new("next_day", DataType::Date32, true)),
                config_options: Arc::new(config),
            }
        };

        // 'ſu' (Latin small letter long s) folds to 'SU' (Sunday -> 2024-01-07, epoch 19729)
        for ansi in [true, false] {
            let res = func
                .invoke_with_args(make_args(
                    ScalarValue::Date32(Some(monday)),
                    "ſu",
                    ansi,
                ))
                .unwrap();
            let ColumnarValue::Scalar(ScalarValue::Date32(Some(val))) = res else {
                panic!("expected Date32 scalar");
            };
            assert_eq!(
                val, 19729,
                "expected Sunday 2024-01-07 for 'ſu' (ansi={ansi})"
            );
        }

        // 'frı' (Latin small letter dotless i) folds to 'FRI' (Friday -> 2024-01-05, epoch 19727)
        for ansi in [true, false] {
            let res = func
                .invoke_with_args(make_args(
                    ScalarValue::Date32(Some(monday)),
                    "frı",
                    ansi,
                ))
                .unwrap();
            let ColumnarValue::Scalar(ScalarValue::Date32(Some(val))) = res else {
                panic!("expected Date32 scalar");
            };
            assert_eq!(
                val, 19727,
                "expected Friday 2024-01-05 for 'frı' (ansi={ansi})"
            );
        }
    }
}
