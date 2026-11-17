// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Date32Array, Int32Array};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Date32Type, Field, FieldRef, Int32Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature,
};
use datafusion_macros::user_doc;

use super::date_part_name::date_signature;

/// Spark `DayOfWeek`: Sunday = 1, Monday = 2, ..., Saturday = 7.
///
/// Spark computes `LocalDate.ofEpochDay(days).getDayOfWeek.plus(1).getValue`,
/// which is defined for every `Int`. Epoch day 0 (1970-01-01) is a Thursday, so
/// `+ 4` moves Sunday to 0 before the shift to 1-based. Widening to `i64` first
/// keeps `i32::MAX` from overflowing.
#[inline]
pub(super) fn day_of_week(days: i32) -> i32 {
    ((i64::from(days) + 4).rem_euclid(7) + 1) as i32
}

/// Spark `WeekDay`: Monday = 0, Tuesday = 1, ..., Sunday = 6.
///
/// Spark computes `LocalDate.ofEpochDay(days).getDayOfWeek.ordinal()`.
#[inline]
pub(super) fn week_day(days: i32) -> i32 {
    (i64::from(days) + 3).rem_euclid(7) as i32
}

/// Applies `kernel` to every valid date.
///
/// `unary` evaluates every slot and vectorizes, `unary_opt` visits only valid
/// slots but costs more per element, so switch on null density.
#[inline]
fn map_dates<F: Fn(i32) -> i32 + Copy>(dates: &Date32Array, kernel: F) -> Int32Array {
    if dates.null_count() * 2 <= dates.len() {
        dates.unary::<_, Int32Type>(kernel)
    } else {
        dates.unary_opt::<_, Int32Type>(|d| Some(kernel(d)))
    }
}

/// Maps each epoch day in `arg` (a `Date32` scalar or array, possibly
/// dictionary encoded) to an `Int32` with `kernel`. NULL dates give NULL.
///
/// The value is computed straight from the epoch day rather than through a
/// calendar date, so it is defined for the whole `i32` domain as in Spark.
pub(super) fn epoch_day_field<F: Fn(i32) -> i32 + Copy>(
    name: &str,
    arg: ColumnarValue,
    kernel: F,
) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Date32(days)) => {
            Ok(ColumnarValue::Scalar(ScalarValue::Int32(days.map(kernel))))
        }
        ColumnarValue::Scalar(scalar) if scalar.is_null() => {
            Ok(ColumnarValue::Scalar(ScalarValue::Int32(None)))
        }
        ColumnarValue::Scalar(scalar) => {
            let array = scalar.to_array_of_size(1)?;
            let result = epoch_day_array(name, &array, kernel)?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        }
        ColumnarValue::Array(array) => {
            Ok(ColumnarValue::Array(epoch_day_array(name, &array, kernel)?))
        }
    }
}

fn epoch_day_array<F: Fn(i32) -> i32 + Copy>(
    name: &str,
    array: &ArrayRef,
    kernel: F,
) -> Result<ArrayRef> {
    // The date signature also admits Date64, which holds milliseconds since the
    // epoch. Arrow's cast turns it into epoch days.
    if array.data_type() == &DataType::Date64 {
        return epoch_day_array(name, &cast(array, &DataType::Date32)?, kernel);
    }
    // A dictionary-encoded date column: map the dictionary values, rewrap the
    // keys, then unpack to a plain Int32 array, so the work is proportional to
    // the cardinality rather than the row count.
    if let DataType::Dictionary(_, _) = array.data_type() {
        let dict = array.as_any_dictionary();
        let values = match dict.values().data_type() {
            DataType::Date64 => cast(dict.values(), &DataType::Date32)?,
            _ => Arc::clone(dict.values()),
        };
        let Some(values) = values.as_primitive_opt::<Date32Type>() else {
            return exec_err!("`{name}` expects a date, got {}", array.data_type());
        };
        let mapped = Arc::new(map_dates(values, kernel)) as ArrayRef;
        return Ok(cast(&dict.with_values(mapped), &DataType::Int32)?);
    }
    let Some(dates) = array.as_primitive_opt::<Date32Type>() else {
        return exec_err!("`{name}` expects a date, got {}", array.data_type());
    };
    Ok(Arc::new(map_dates(dates, kernel)))
}

/// Spark-compatible `dayofweek` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#dayofweek>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the day of the week as an INT from 1 (Sunday) to 7 (Saturday). \
        This differs from `weekday`, which counts from 0 (Monday) to 6 (Sunday), and from \
        DataFusion's `date_part('dow', ...)`, which counts from 0 (Sunday) to 6 (Saturday). \
        DataFusion has no built-in `dayofweek`. \
        Strings are implicitly cast to DATE. A malformed string is an error, even with ANSI mode off \
        where Spark returns NULL. A string that Spark reads as a partial date, such as '2009' or \
        '2009-07', is also an error, where Spark uses the first day of that year or month. \
        Timestamps are cast to DATE first, so the time of day is ignored. A timestamp with a time zone \
        is converted in its own time zone, not the session time zone that Spark uses. \
        Returns NULL if the input is NULL.",
    syntax_example = "dayofweek(date)",
    sql_example = r#"```sql
> SELECT dayofweek('2009-07-30'::DATE);
5
> SELECT dayofweek('2009-07-30');
5
> SELECT dayofweek(arrow_cast('2024-01-07 23:59:59', 'Timestamp(Microsecond, None)'));
1
> SELECT dayofweek(NULL::DATE);
NULL
```"#,
    argument(
        name = "date",
        description = "DATE, TIMESTAMP or string expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDayOfWeek {
    signature: Signature,
}

impl Default for SparkDayOfWeek {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDayOfWeek {
    pub fn new() -> Self {
        Self {
            signature: date_signature(),
        }
    }
}

impl ScalarUDFImpl for SparkDayOfWeek {
    fn name(&self) -> &str {
        "dayofweek"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, nullable)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        epoch_day_field(self.name(), arg, day_of_week)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Date64Array, DictionaryArray};
    use datafusion_common::config::ConfigOptions;

    fn invoke(arg: ColumnarValue, rows: usize) -> Result<ColumnarValue> {
        let data_type = arg.data_type();
        SparkDayOfWeek::new().invoke_with_args(ScalarFunctionArgs {
            args: vec![arg],
            arg_fields: vec![Arc::new(Field::new("d", data_type, true))],
            number_rows: rows,
            return_field: Arc::new(Field::new("dayofweek", DataType::Int32, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn invoke_array(array: ArrayRef) -> Int32Array {
        let rows = array.len();
        let result = invoke(ColumnarValue::Array(array), rows)
            .unwrap()
            .to_array(rows)
            .unwrap();
        result.as_primitive::<Int32Type>().clone()
    }

    #[test]
    fn return_field_nullability_matches_input() {
        let func = SparkDayOfWeek::new();
        for nullable in [false, true] {
            let arg = Arc::new(Field::new("arg", DataType::Date32, nullable));
            let out = func
                .return_field_from_args(ReturnFieldArgs {
                    arg_fields: &[arg],
                    scalar_arguments: &[None],
                })
                .unwrap();
            assert_eq!(out.data_type(), &DataType::Int32);
            assert_eq!(out.is_nullable(), nullable);
        }
    }

    /// 1970-01-01 is a Thursday, so the week either side of the epoch pins
    /// both numberings.
    #[test]
    fn known_weekdays_around_the_epoch() {
        // days -4..=3 spans Sunday 1969-12-28 through Sunday 1970-01-04.
        let array: ArrayRef = Arc::new(Date32Array::from((-4..=3).collect::<Vec<_>>()));
        assert_eq!(
            invoke_array(array).values(),
            // Sun Mon Tue Wed Thu Fri Sat Sun
            &[1, 2, 3, 4, 5, 6, 7, 1]
        );
        let weekdays: Vec<i32> = (-4..=3).map(week_day).collect();
        assert_eq!(weekdays, [6, 0, 1, 2, 3, 4, 5, 6]);
    }

    /// Nulls pass through in place, at both null densities `map_dates` handles.
    #[test]
    fn nulls_are_preserved() {
        let sparse: ArrayRef =
            Arc::new(Date32Array::from(vec![Some(0), None, Some(1), Some(2)]));
        let out = invoke_array(sparse);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(5), None, Some(6), Some(7)]
        );

        let dense: ArrayRef =
            Arc::new(Date32Array::from(vec![None, None, Some(1), None]));
        let out = invoke_array(dense);
        assert_eq!(out.iter().collect::<Vec<_>>(), [None, None, Some(6), None]);
    }

    /// The whole `i32` domain works, as `LocalDate.ofEpochDay` does in Spark.
    /// These are outside chrono's representable range.
    #[test]
    fn full_i32_domain() {
        let array: ArrayRef = Arc::new(Date32Array::from(vec![i32::MIN, i32::MAX]));
        // Verified against Spark 4.2.0:
        // dayofweek(date_from_unix_date(-2147483648)) = 3 (-5877641-06-23)
        // dayofweek(date_from_unix_date(2147483647)) = 6 (+5881580-07-11)
        // weekday of the same two dates = 1 and 4
        assert_eq!(invoke_array(array).values(), &[3, 6]);
        assert_eq!([week_day(i32::MIN), week_day(i32::MAX)], [1, 4]);
    }

    /// The two numberings agree across a full 400-year Gregorian cycle.
    #[test]
    fn numberings_agree_over_a_full_cycle() {
        for days in -146_097..=146_097 {
            assert_eq!((day_of_week(days) + 5) % 7, week_day(days), "day {days}");
        }
    }

    /// A dictionary-encoded date column unpacks to a plain Int32 result.
    #[test]
    fn dictionary_input_unpacks() {
        let values =
            Arc::new(Date32Array::from(vec![Some(0), Some(1), None])) as ArrayRef;
        let keys = Int32Array::from(vec![Some(0), Some(1), Some(0), None, Some(2)]);
        let dict = DictionaryArray::<Int32Type>::new(keys, values);
        let out = invoke_array(Arc::new(dict));
        assert_eq!(out.data_type(), &DataType::Int32);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(5), Some(6), Some(5), None, None]
        );
    }

    /// A dictionary of Date64 values is cast to epoch days before mapping.
    /// SQL never reaches this path, because coercion unpacks the dictionary
    /// to its value type first, so only a direct call covers it.
    #[test]
    fn date64_dictionary_input() {
        let values = Arc::new(Date64Array::from(vec![Some(0), Some(-86_400_000), None]))
            as ArrayRef;
        let keys = Int32Array::from(vec![Some(0), Some(1), None, Some(2)]);
        let dict = DictionaryArray::<Int32Type>::new(keys, values);
        let out = invoke_array(Arc::new(dict));
        assert_eq!(out.data_type(), &DataType::Int32);
        assert_eq!(
            out.iter().collect::<Vec<_>>(),
            [Some(5), Some(4), None, None]
        );
    }

    #[test]
    fn rejects_non_date_input() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let err = invoke(ColumnarValue::Array(array), 1).unwrap_err();
        assert!(err.to_string().contains("expects a date"), "{err}");
    }

    #[test]
    fn scalar_input() -> Result<()> {
        match invoke(ColumnarValue::Scalar(ScalarValue::Date32(Some(0))), 1)? {
            ColumnarValue::Scalar(ScalarValue::Int32(Some(5))) => {}
            other => panic!("unexpected result: {other:?}"),
        }
        match invoke(ColumnarValue::Scalar(ScalarValue::Date32(None)), 1)? {
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => {}
            other => panic!("unexpected result: {other:?}"),
        }
        match invoke(ColumnarValue::Scalar(ScalarValue::Null), 1)? {
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => {}
            other => panic!("unexpected result: {other:?}"),
        }
        Ok(())
    }
}
