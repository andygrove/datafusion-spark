// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, Date32Array, Date32Builder, new_empty_array};
use arrow::datatypes::{DataType, FieldRef};
use datafusion_common::cast::as_int32_array;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use super::date_part_name::int32_coercion;
use crate::function::nullability::{SparkNullability, spark_nullability};

const ARITHMETIC_OVERFLOW_ERROR: &str = r#"[ARITHMETIC_OVERFLOW] overflow. If necessary set "spark.sql.ansi.enabled" to "false" to bypass this error. SQLSTATE: 22003"#;

/// Spark's `MakeDate` is null-intolerant. It is always nullable when ANSI mode is off,
/// because an invalid date returns NULL, and nullable when an argument is when ANSI mode is
/// on. ANSI mode is not known in `return_field_from_args`, so the result is always nullable
/// here.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant_always_nullable();

/// Spark-compatible `make_date` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#make_date>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Creates a date from `year`, `month` and `day` fields. Years from -999999999 to 999999999 are accepted (year 0 is 1 BC), as long as \
        the date fits the DATE range. When the fields do not form a valid date (for example \
        month 13 or February 29 in a non-leap year), the function returns NULL, or raises \
        DATETIME_FIELD_OUT_OF_BOUNDS when `datafusion.execution.enable_ansi_mode` is true. A \
        valid date outside the DATE range (before -5877641-06-23 or after +5881580-07-11) raises \
        ARITHMETIC_OVERFLOW in both modes, as in Spark. Other numeric and string inputs are cast \
        to INT first, which truncates fractions and raises an error if the value is out of the \
        INT range or is not a valid number. These errors match Spark with ANSI mode enabled. \
        With ANSI mode disabled, Spark wraps an out-of-range BIGINT or DECIMAL, truncates a \
        string with a fractional part, returns NULL for a malformed string or a floating-point \
        value out of the INT range, and casts NaN to 0, while this function still errors (see \
        https://github.com/apache/datafusion-spark/issues/88). Returns NULL if any input is \
        NULL. DataFusion's built-in `make_date` differs: it raises an error for an invalid \
        date regardless of ANSI mode, accepts only integer and string arguments, and is \
        limited to years -262143 to 262142.",
    syntax_example = "make_date(year, month, day)",
    sql_example = r#"```sql
> SELECT make_date(2013, 7, 15);
2013-07-15
> SELECT make_date(2019, 7, NULL);
NULL
> SELECT make_date(2023, 2, 29);
NULL
```"#,
    argument(
        name = "year",
        description = "The year, from -999999999 to 999999999, as an INT. Other integer, floating-point, decimal and string inputs are cast to INT. NULL returns NULL."
    ),
    argument(
        name = "month",
        description = "The month of the year, from 1 (January) to 12 (December), as an INT. NULL returns NULL."
    ),
    argument(
        name = "day",
        description = "The day of the month, from 1 to 31, as an INT. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMakeDate {
    signature: Signature,
}

impl Default for SparkMakeDate {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMakeDate {
    pub fn new() -> Self {
        // Spark declares the arguments as INT.
        Self {
            signature: Signature::coercible(
                vec![int32_coercion(), int32_coercion(), int32_coercion()],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkMakeDate {
    fn name(&self) -> &str {
        "make_date"
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
        let fail_on_error = args.config_options.execution.enable_ansi_mode;
        let number_rows = args.number_rows;
        let [year, month, day] = take_function_args(self.name(), args.args)?;

        if let (
            ColumnarValue::Scalar(year),
            ColumnarValue::Scalar(month),
            ColumnarValue::Scalar(day),
        ) = (&year, &month, &day)
        {
            // Spark does not evaluate the expression when there are no rows, so an
            // invalid constant date must not raise an error on an empty batch.
            if number_rows == 0 {
                return Ok(ColumnarValue::Array(new_empty_array(&DataType::Date32)));
            }
            let days = match (year, month, day) {
                (
                    ScalarValue::Int32(Some(y)),
                    ScalarValue::Int32(Some(m)),
                    ScalarValue::Int32(Some(d)),
                ) => make_date(*y, *m, *d, fail_on_error)?,
                (
                    ScalarValue::Int32(_) | ScalarValue::Null,
                    ScalarValue::Int32(_) | ScalarValue::Null,
                    ScalarValue::Int32(_) | ScalarValue::Null,
                ) => None,
                _ => {
                    return internal_err!(
                        "make_date expects INT arguments, got {}, {}, {}",
                        year.data_type(),
                        month.data_type(),
                        day.data_type()
                    );
                }
            };
            return Ok(ColumnarValue::Scalar(ScalarValue::Date32(days)));
        }

        let year = year.into_array(number_rows)?;
        let month = month.into_array(number_rows)?;
        let day = day.into_array(number_rows)?;
        let years = as_int32_array(&year)?;
        let months = as_int32_array(&month)?;
        let days = as_int32_array(&day)?;

        let mut builder = Date32Builder::with_capacity(number_rows);
        for i in 0..number_rows {
            if years.is_null(i) || months.is_null(i) || days.is_null(i) {
                builder.append_null();
                continue;
            }
            builder.append_option(make_date(
                years.value(i),
                months.value(i),
                days.value(i),
                fail_on_error,
            )?);
        }
        let result: Date32Array = builder.finish();
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

/// Computes the DATE (days since 1970-01-01) for `year`, `month` and `day`, as
/// Spark does with `localDateToDays(LocalDate.of(year, month, day))`.
///
/// Returns `Ok(None)` for an invalid date when `fail_on_error` is false, and an
/// error otherwise. A valid date whose day count does not fit in an INT raises
/// `ARITHMETIC_OVERFLOW` regardless of `fail_on_error`, because Spark only
/// catches the `DateTimeException` from `LocalDate.of` and not the overflow of
/// `Math.toIntExact`.
fn make_date(
    year: i32,
    month: i32,
    day: i32,
    fail_on_error: bool,
) -> Result<Option<i32>> {
    if let Some(message) = invalid_date_message(year, month, day) {
        if fail_on_error {
            return exec_err!(
                "[DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION] {message}. If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error. SQLSTATE: 22023"
            );
        }
        return Ok(None);
    }
    match i32::try_from(days_from_civil(year, month, day)) {
        Ok(days) => Ok(Some(days)),
        Err(_) => exec_err!("{ARITHMETIC_OVERFLOW_ERROR}"),
    }
}

/// Returns the message `java.time.LocalDate.of` raises for an invalid date, or
/// `None` if the date is valid. The checks run in the same order as in Java:
/// year range, month range, day range, then the day against the month length.
fn invalid_date_message(year: i32, month: i32, day: i32) -> Option<String> {
    const MONTH_NAMES: [&str; 12] = [
        "JANUARY",
        "FEBRUARY",
        "MARCH",
        "APRIL",
        "MAY",
        "JUNE",
        "JULY",
        "AUGUST",
        "SEPTEMBER",
        "OCTOBER",
        "NOVEMBER",
        "DECEMBER",
    ];
    if !(-999_999_999..=999_999_999).contains(&year) {
        return Some(format!(
            "Invalid value for Year (valid values -999999999 - 999999999): {year}"
        ));
    }
    if !(1..=12).contains(&month) {
        return Some(format!(
            "Invalid value for MonthOfYear (valid values 1 - 12): {month}"
        ));
    }
    if !(1..=31).contains(&day) {
        return Some(format!(
            "Invalid value for DayOfMonth (valid values 1 - 28/31): {day}"
        ));
    }
    if day > days_in_month(year, month) {
        if month == 2 && day == 29 {
            return Some(format!(
                "Invalid date 'February 29' as '{year}' is not a leap year"
            ));
        }
        return Some(format!(
            "Invalid date '{} {day}'",
            MONTH_NAMES[(month - 1) as usize]
        ));
    }
    None
}

/// Proleptic Gregorian leap year rule, as used by `java.time`.
fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 for a valid proleptic Gregorian date, computed in
/// `i64` so that the full `java.time` year range does not overflow.
/// Based on Howard Hinnant's `days_from_civil` algorithm.
fn days_from_civil(year: i32, month: i32, day: i32) -> i64 {
    let (year, month, day) = (i64::from(year), i64::from(month), i64::from(day));
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year.rem_euclid(400);
    let shifted_month = (month + 9) % 12; // March = 0, ..., February = 11
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era =
        year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(
        args: Vec<ColumnarValue>,
        number_rows: usize,
        ansi: bool,
    ) -> Result<ColumnarValue> {
        let mut config = ConfigOptions::default();
        config.execution.enable_ansi_mode = ansi;
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(i, a)| Arc::new(Field::new(format!("a{i}"), a.data_type(), true)))
            .collect();
        SparkMakeDate::new().invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new("make_date", DataType::Date32, true)),
            config_options: Arc::new(config),
        })
    }

    fn scalar(v: Option<i32>) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Int32(v))
    }

    fn array(v: Vec<Option<i32>>) -> ColumnarValue {
        ColumnarValue::Array(Arc::new(Int32Array::from(v)))
    }

    fn ok(y: i32, m: i32, d: i32) -> Option<i32> {
        make_date(y, m, d, false).unwrap()
    }

    #[test]
    fn test_make_date_valid() {
        assert_eq!(ok(1970, 1, 1), Some(0));
        assert_eq!(ok(1970, 1, 2), Some(1));
        assert_eq!(ok(1969, 12, 31), Some(-1));
        assert_eq!(ok(2000, 2, 29), Some(11_016));
        assert_eq!(ok(2004, 2, 29), Some(12_477));
        assert_eq!(ok(2013, 7, 15), Some(15_901));
    }

    #[test]
    fn test_make_date_invalid_month() {
        assert_eq!(ok(2023, 0, 15), None);
        assert_eq!(ok(2023, 13, 15), None);
        assert_eq!(ok(2023, -1, 15), None);
    }

    #[test]
    fn test_make_date_invalid_day() {
        assert_eq!(ok(2023, 6, 0), None);
        assert_eq!(ok(2023, 6, 32), None);
        assert_eq!(ok(2023, 6, -1), None);
    }

    #[test]
    fn test_make_date_invalid_dates() {
        assert_eq!(ok(2023, 2, 30), None);
        assert_eq!(ok(2023, 2, 29), None);
        // 1900 is not a leap year (divisible by 100 but not 400)
        assert_eq!(ok(1900, 2, 29), None);
        assert_eq!(ok(2100, 2, 29), None);
        assert_eq!(ok(2023, 4, 31), None);
        assert_eq!(ok(1_000_000_000, 1, 1), None);
        assert_eq!(ok(-1_000_000_000, 1, 1), None);
    }

    #[test]
    fn test_make_date_wide_year_range() {
        assert_eq!(ok(0, 1, 1), Some(-719_528));
        assert_eq!(ok(-1, 1, 1), Some(-719_893));
        assert_eq!(ok(300_000, 6, 15), Some(108_853_388));
        assert_eq!(ok(300_000, 2, 29), Some(108_853_281));
        assert_eq!(ok(5_881_580, 7, 11), Some(i32::MAX));
        assert_eq!(ok(-5_877_641, 6, 23), Some(i32::MIN));
    }

    #[test]
    fn test_make_date_overflow_errors_in_both_modes() {
        for ansi in [false, true] {
            for (y, m, d) in [
                (5_881_580, 7, 12),
                (-5_877_641, 6, 22),
                (999_999_999, 12, 31),
            ] {
                let err = make_date(y, m, d, ansi).unwrap_err().to_string();
                assert!(err.contains("[ARITHMETIC_OVERFLOW]"), "{err}");
            }
        }
    }

    #[test]
    fn test_make_date_ansi_messages() {
        let cases = [
            (
                2023,
                2,
                29,
                "Invalid date 'February 29' as '2023' is not a leap year",
            ),
            (2023, 4, 31, "Invalid date 'APRIL 31'"),
            (
                2023,
                13,
                40,
                "Invalid value for MonthOfYear (valid values 1 - 12): 13",
            ),
            (
                2023,
                6,
                0,
                "Invalid value for DayOfMonth (valid values 1 - 28/31): 0",
            ),
            (
                1_000_000_000,
                13,
                40,
                "Invalid value for Year (valid values -999999999 - 999999999): 1000000000",
            ),
        ];
        for (y, m, d, message) in cases {
            let err = make_date(y, m, d, true).unwrap_err().to_string();
            assert!(
                err.contains(&format!(
                    "[DATETIME_FIELD_OUT_OF_BOUNDS.WITH_SUGGESTION] {message}. If necessary"
                )),
                "{err}"
            );
            assert!(err.ends_with("SQLSTATE: 22023"), "{err}");
        }
    }

    #[test]
    fn test_make_date_scalar() -> Result<()> {
        let result = invoke(
            vec![scalar(Some(2013)), scalar(Some(7)), scalar(Some(15))],
            1,
            false,
        )?;
        let ColumnarValue::Scalar(ScalarValue::Date32(Some(15_901))) = result else {
            panic!("unexpected {result:?}");
        };
        let result = invoke(
            vec![scalar(Some(2023)), scalar(Some(2)), scalar(Some(29))],
            1,
            false,
        )?;
        let ColumnarValue::Scalar(ScalarValue::Date32(None)) = result else {
            panic!("unexpected {result:?}");
        };
        let result = invoke(
            vec![
                scalar(Some(2019)),
                scalar(Some(7)),
                ColumnarValue::Scalar(ScalarValue::Null),
            ],
            1,
            true,
        )?;
        let ColumnarValue::Scalar(ScalarValue::Date32(None)) = result else {
            panic!("unexpected {result:?}");
        };
        assert!(
            invoke(
                vec![scalar(Some(2023)), scalar(Some(2)), scalar(Some(29))],
                1,
                true
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn test_make_date_scalar_empty_batch_does_not_validate() -> Result<()> {
        let result = invoke(
            vec![scalar(Some(2024)), scalar(Some(13)), scalar(Some(1))],
            0,
            true,
        )?;
        let ColumnarValue::Array(array) = result else {
            panic!("expected an empty array");
        };
        assert_eq!(array.data_type(), &DataType::Date32);
        assert!(array.is_empty());
        Ok(())
    }

    #[test]
    fn test_make_date_array() -> Result<()> {
        let years = array(vec![Some(2024), Some(2023), None, Some(2020), Some(2023)]);
        let result = invoke(
            vec![
                years,
                scalar(Some(2)),
                array(vec![Some(29), Some(29), Some(1), Some(29), None]),
            ],
            5,
            false,
        )?
        .into_array(5)?;
        let result = result.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(19_782), None, None, Some(18_321), None]
        );
        Ok(())
    }

    #[test]
    fn test_make_date_array_ansi() -> Result<()> {
        // NULL inputs are skipped before validation, so an invalid month next to a
        // NULL year does not raise an error.
        let result = invoke(
            vec![
                array(vec![Some(2024), None]),
                array(vec![Some(1), Some(13)]),
                scalar(Some(31)),
            ],
            2,
            true,
        )?
        .into_array(2)?;
        let result = result.as_any().downcast_ref::<Date32Array>().unwrap();
        assert_eq!(result.iter().collect::<Vec<_>>(), vec![Some(19_753), None]);

        let err = invoke(
            vec![
                array(vec![Some(2024), Some(2023)]),
                scalar(Some(2)),
                scalar(Some(29)),
            ],
            2,
            true,
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("Invalid date 'February 29' as '2023'"),
            "{err}"
        );
        Ok(())
    }
}
