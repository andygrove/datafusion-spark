// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use crate::function::nullability::{SparkNullability, spark_nullability};
use arrow::array::{
    Array, ArrayRef, AsArray, DurationMicrosecondBuilder, PrimitiveArray,
};
use arrow::datatypes::TimeUnit::Microsecond;
use arrow::datatypes::{DataType, FieldRef, Float64Type, Int32Type};
use datafusion_common::types::{NativeType, logical_float64, logical_int32};
use datafusion_common::{
    DataFusionError, Result, ScalarValue, internal_err, plan_datafusion_err,
};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark's `MakeDTInterval` is a null-intolerant `QuaternaryExpression` that is nullable
/// when an argument is, but here an overflow returns NULL (#20), so the result is always
/// nullable.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant_always_nullable();

/// Spark-compatible `make_dt_interval` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#make_dt_interval>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns a day-time interval built from days, hours, minutes and seconds. \
        DataFusion has no built-in equivalent. All arguments are optional from the right and default to 0, \
        so `make_dt_interval()` is a zero interval. Components can be negative and are added together, \
        so `make_dt_interval(1, -24)` is zero. The result is a `Duration(Microsecond)` rather than an Arrow interval, \
        because Arrow's day-time interval only has millisecond precision while Spark's has microsecond precision. \
        Returns NULL if any argument is NULL or `secs` is NaN or infinite. Overflow returns NULL, where Spark raises an error ([#20](https://github.com/apache/datafusion-spark/issues/20)). \
        Unlike Spark, where `secs` is a DECIMAL(18, 6), `secs` is cast to DOUBLE and the fraction is rounded to microseconds.",
    syntax_example = "make_dt_interval([days[, hours[, mins[, secs]]]])",
    sql_example = r#"```sql
> SELECT make_dt_interval(1, 12, 30, 1.001001);
1 days 12 hours 30 mins 1.001001 secs
> SELECT make_dt_interval(2);
2 days 0 hours 0 mins 0.000000 secs
> SELECT make_dt_interval(100, NULL, 3);
NULL
```"#,
    argument(
        name = "days",
        description = "Integer number of days, cast to INT. NULL returns NULL."
    ),
    argument(
        name = "hours",
        description = "Integer number of hours, cast to INT. NULL returns NULL."
    ),
    argument(
        name = "mins",
        description = "Integer number of minutes, cast to INT. NULL returns NULL."
    ),
    argument(
        name = "secs",
        description = "Numeric number of seconds, which may have a fraction, cast to DOUBLE. NULL, NaN or an infinite value returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMakeDtInterval {
    signature: Signature,
}

impl Default for SparkMakeDtInterval {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMakeDtInterval {
    pub fn new() -> Self {
        let int32 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int32()),
            vec![TypeSignatureClass::Integer],
            NativeType::Int32,
        );

        let float64 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_float64()),
            vec![TypeSignatureClass::Numeric],
            NativeType::Float64,
        );

        let variants = vec![
            TypeSignature::Nullary,
            // (days)
            TypeSignature::Coercible(vec![int32.clone()]),
            // (days, hours)
            TypeSignature::Coercible(vec![int32.clone(), int32.clone()]),
            // (days, hours, minutes)
            TypeSignature::Coercible(vec![int32.clone(), int32.clone(), int32.clone()]),
            // (days, hours, minutes, seconds)
            TypeSignature::Coercible(vec![
                int32.clone(),
                int32.clone(),
                int32.clone(),
                float64,
            ]),
        ];

        Self {
            signature: Signature::one_of(variants, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkMakeDtInterval {
    fn name(&self) -> &str {
        "make_dt_interval"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Note the return type is `DataType::Duration(TimeUnit::Microsecond)` and not `DataType::Interval(DayTime)` as you might expect.
    /// This is because `DataType::Interval(DayTime)` has precision only to the millisecond, whilst Spark's `DayTimeIntervalType` has
    /// precision to the microsecond. We use `DataType::Duration(TimeUnit::Microsecond)` in order to not lose any precision. See the
    /// [Sail compatibility doc] for reference.
    ///
    /// [Sail compatibility doc]: https://github.com/lakehq/sail/blob/dc5368daa24d40a7758a299e1ba8fc985cb29108/docs/guide/dataframe/data-types/compatibility.md?plain=1#L260
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(NULLABILITY.return_field(
            self.name(),
            DataType::Duration(Microsecond),
            args.arg_fields,
        ))
    }

    spark_nullability!(NULLABILITY);

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.is_empty() {
            return Ok(ColumnarValue::Scalar(ScalarValue::DurationMicrosecond(
                Some(0),
            )));
        }
        if args.args.len() > 4 {
            return Err(DataFusionError::Execution(format!(
                "make_dt_interval expects between 0 and 4 arguments, got {}",
                args.args.len()
            )));
        }
        make_scalar_function(make_dt_interval_kernel, vec![])(&args.args)
    }
}

fn make_dt_interval_kernel(args: &[ArrayRef]) -> Result<ArrayRef, DataFusionError> {
    let n_rows = args[0].len();
    let days = args[0]
        .as_primitive_opt::<Int32Type>()
        .ok_or_else(|| plan_datafusion_err!("make_dt_interval arg[0] must be Int32"))?;
    let hours: Option<&PrimitiveArray<Int32Type>> = args
        .get(1)
        .map(|a| {
            a.as_primitive_opt::<Int32Type>().ok_or_else(|| {
                plan_datafusion_err!("make_dt_interval arg[1] must be Int32")
            })
        })
        .transpose()?;
    let mins: Option<&PrimitiveArray<Int32Type>> = args
        .get(2)
        .map(|a| {
            a.as_primitive_opt::<Int32Type>().ok_or_else(|| {
                plan_datafusion_err!("make_dt_interval arg[2] must be Int32")
            })
        })
        .transpose()?;
    let secs: Option<&PrimitiveArray<Float64Type>> = args
        .get(3)
        .map(|a| {
            a.as_primitive_opt::<Float64Type>().ok_or_else(|| {
                plan_datafusion_err!("make_dt_interval arg[3] must be Float64")
            })
        })
        .transpose()?;
    let mut builder = DurationMicrosecondBuilder::with_capacity(n_rows);

    for i in 0..n_rows {
        // if one column is NULL → result NULL
        let any_null_present = days.is_null(i)
            || hours.as_ref().is_some_and(|a| a.is_null(i))
            || mins.as_ref().is_some_and(|a| a.is_null(i))
            || secs
                .as_ref()
                .is_some_and(|a| a.is_null(i) || !a.value(i).is_finite());

        if any_null_present {
            builder.append_null();
            continue;
        }

        // default values 0 or 0.0
        let d = days.value(i);
        let h = hours.as_ref().map_or(0, |a| a.value(i));
        let mi = mins.as_ref().map_or(0, |a| a.value(i));
        let s = secs.as_ref().map_or(0.0, |a| a.value(i));

        match make_interval_dt_nano(d, h, mi, s) {
            Some(v) => builder.append_value(v),
            None => builder.append_null(),
        }
    }

    Ok(Arc::new(builder.finish()))
}
fn make_interval_dt_nano(day: i32, hour: i32, min: i32, sec: f64) -> Option<i64> {
    const MICROS_PER_SEC: i64 = 1_000_000;
    const MICROS_PER_MINUTE: i64 = 60 * MICROS_PER_SEC;
    const MICROS_PER_HOUR: i64 = 60 * MICROS_PER_MINUTE;
    const MICROS_PER_DAY: i64 = 24 * MICROS_PER_HOUR;
    const MAX_SECS_ABS: f64 = (1i64 << 53) as f64;

    if !sec.is_finite() || sec <= -MAX_SECS_ABS || sec >= MAX_SECS_ABS {
        return None;
    }

    // Step 1: Convert seconds (whole and fractional) to microseconds
    let mut sec_whole: i64 = sec.trunc() as i64;
    let sec_frac: f64 = sec - (sec_whole as f64);
    let mut frac_us: i64 = (sec_frac * (MICROS_PER_SEC as f64)).round() as i64;

    if frac_us >= MICROS_PER_SEC {
        frac_us -= MICROS_PER_SEC;
        sec_whole = sec_whole.checked_add(1)?;
    } else if frac_us <= -MICROS_PER_SEC {
        frac_us += MICROS_PER_SEC;
        sec_whole = sec_whole.checked_sub(1)?;
    }

    let sec_us: i64 = sec_whole
        .checked_mul(MICROS_PER_SEC)?
        .checked_add(frac_us)?;

    // Step 2: Accumulate components sequentially in microseconds, matching Spark's
    // exact IntervalUtils.makeDayTimeInterval evaluation order:
    //   var micros = secs.toUnscaledLong
    //   micros = Math.addExact(micros, Math.multiplyExact(days, MICROS_PER_DAY))
    //   micros = Math.addExact(micros, Math.multiplyExact(hours, MICROS_PER_HOUR))
    //   micros = Math.addExact(micros, Math.multiplyExact(mins, MICROS_PER_MINUTE))
    // Checking each addition and multiplication independently prevents opposite-sign
    // components from cancelling an intermediate overflow.
    let mut total_us = sec_us;

    let day_us = (day as i64).checked_mul(MICROS_PER_DAY)?;
    total_us = total_us.checked_add(day_us)?;

    let hour_us = (hour as i64).checked_mul(MICROS_PER_HOUR)?;
    total_us = total_us.checked_add(hour_us)?;

    let min_us = (min as i64).checked_mul(MICROS_PER_MINUTE)?;
    total_us = total_us.checked_add(min_us)?;

    Some(total_us)
}

#[cfg(test)]
mod tests {

    use arrow::array::{DurationMicrosecondArray, Float64Array, Int32Array};
    use arrow::datatypes::DataType::Duration;
    use arrow::datatypes::TimeUnit::Microsecond;
    use datafusion_common::internal_datafusion_err;

    use super::*;
    use arrow::datatypes::Field;

    fn run_make_dt_interval(arrs: Vec<ArrayRef>) -> Result<ArrayRef> {
        make_dt_interval_kernel(&arrs)
    }

    #[test]
    fn nulls_propagate_per_row() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![
            None,
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
        ])) as ArrayRef;

        let hours = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(7),
        ])) as ArrayRef;

        let mins = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(4),
            Some(5),
            Some(6),
            Some(7),
        ])) as ArrayRef;

        let secs = Arc::new(Float64Array::from(vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            None,
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        for i in 0..out.len() {
            assert!(out.is_null(i), "row {i} should be NULL");
        }
        Ok(())
    }

    #[test]
    fn return_field_respects_nullability() -> Result<()> {
        let udf = SparkMakeDtInterval::new();

        // All nullable inputs -> nullable output
        let arg_fields = vec![
            Arc::new(Field::new("days", DataType::Int32, true)),
            Arc::new(Field::new("hours", DataType::Int32, true)),
            Arc::new(Field::new("mins", DataType::Int32, true)),
            Arc::new(Field::new("secs", DataType::Float64, true)),
        ];

        let out = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None, None, None, None],
        })?;
        assert!(out.is_nullable());
        assert_eq!(out.data_type(), &Duration(Microsecond));

        // Non-nullable inputs -> nullable output, because an overflow returns NULL (#20)
        let non_nullable_arg_fields = vec![
            Arc::new(Field::new("days", DataType::Int32, false)),
            Arc::new(Field::new("hours", DataType::Int32, false)),
            Arc::new(Field::new("mins", DataType::Int32, false)),
            Arc::new(Field::new("secs", DataType::Float64, false)),
        ];

        let out = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &non_nullable_arg_fields,
            scalar_arguments: &[None, None, None, None],
        })?;
        assert!(out.is_nullable());

        // Non-finite secs scalar is nullable too
        let scalar_values =
            [None, None, None, Some(ScalarValue::Float64(Some(f64::NAN)))];
        let scalar_refs = scalar_values.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        let out = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &non_nullable_arg_fields,
            scalar_arguments: &scalar_refs,
        })?;
        assert!(out.is_nullable());

        // A zero-arg call (defaults) is nullable too, as the rule ignores the arguments
        let out = udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[],
            scalar_arguments: &[],
        })?;
        assert!(out.is_nullable());

        Ok(())
    }

    #[test]
    fn error_months_overflow_should_be_null() -> Result<()> {
        // months = year*12 + month → NULL

        let days = Arc::new(Int32Array::from(vec![Some(i32::MAX)])) as ArrayRef;

        let hours = Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef;

        let mins = Arc::new(Int32Array::from(vec![Some(1)])) as ArrayRef;

        let secs = Arc::new(Float64Array::from(vec![Some(1.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        for i in 0..out.len() {
            assert!(out.is_null(i), "row {i} should be NULL");
        }

        Ok(())
    }

    fn invoke_make_dt_interval_with_args(
        args: Vec<ColumnarValue>,
        number_rows: usize,
    ) -> Result<ColumnarValue, DataFusionError> {
        let arg_fields = args
            .iter()
            .map(|arg| Field::new("a", arg.data_type(), true).into())
            .collect::<Vec<_>>();
        let args = ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Field::new("f", Duration(Microsecond), true).into(),
            config_options: Arc::new(Default::default()),
        };
        SparkMakeDtInterval::new().invoke_with_args(args)
    }

    #[test]
    fn zero_args_returns_zero_duration() -> Result<()> {
        let number_rows: usize = 3;

        let res: ColumnarValue = invoke_make_dt_interval_with_args(vec![], number_rows)?;
        let arr = res.into_array(number_rows)?;
        let arr = arr
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(arr.len(), number_rows);
        for i in 0..number_rows {
            assert!(!arr.is_null(i));
            assert_eq!(arr.value(i), 0_i64);
        }
        Ok(())
    }

    #[test]
    fn one_day_minus_24_hours_equals_zero() -> Result<()> {
        let arr_days = Arc::new(Int32Array::from(vec![Some(1), Some(-1)])) as ArrayRef;
        let arr_hours = Arc::new(Int32Array::from(vec![Some(-24), Some(24)])) as ArrayRef;
        let arr_mins = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let arr_secs =
            Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![arr_days, arr_hours, arr_mins, arr_secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 2);
        assert_eq!(out.null_count(), 0);
        assert_eq!(out.value(0), 0_i64);
        assert_eq!(out.value(1), 0_i64);
        Ok(())
    }

    #[test]
    fn one_hour_minus_60_mins_equals_zero() -> Result<()> {
        let arr_days = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let arr_hours = Arc::new(Int32Array::from(vec![Some(-1), Some(1)])) as ArrayRef;
        let arr_mins = Arc::new(Int32Array::from(vec![Some(60), Some(-60)])) as ArrayRef;
        let arr_secs =
            Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![arr_days, arr_hours, arr_mins, arr_secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 2);
        assert_eq!(out.null_count(), 0);
        assert_eq!(out.value(0), 0_i64);
        assert_eq!(out.value(1), 0_i64);
        Ok(())
    }

    #[test]
    fn one_mins_minus_60_secs_equals_zero() -> Result<()> {
        let arr_days = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let arr_hours = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let arr_mins = Arc::new(Int32Array::from(vec![Some(-1), Some(1)])) as ArrayRef;
        let arr_secs =
            Arc::new(Float64Array::from(vec![Some(60.0), Some(-60.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![arr_days, arr_hours, arr_mins, arr_secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 2);
        assert_eq!(out.null_count(), 0);
        assert_eq!(out.value(0), 0_i64);
        assert_eq!(out.value(1), 0_i64);
        Ok(())
    }

    #[test]
    fn frac_carries_up_to_next_second_positive() -> Result<()> {
        // 0.9999995s → 1_000_000 µs (carry a +1s)
        let days = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![
            Some(0.999_999_5),
            Some(0.999_999_4),
        ])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 2);
        assert_eq!(out.value(0), 1_000_000);
        assert_eq!(out.value(1), 999_999);
        Ok(())
    }

    #[test]
    fn frac_carries_down_to_prev_second_negative() -> Result<()> {
        // -0.9999995s → -1_000_000 µs (carry a −1s)
        let days = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0), Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![
            Some(-0.999_999_5),
            Some(-0.999_999_4),
        ])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 2);
        assert_eq!(out.value(0), -1_000_000);
        assert_eq!(out.value(1), -999_999);
        Ok(())
    }

    #[test]
    fn no_more_than_4_params() -> Result<()> {
        let udf = SparkMakeDtInterval::new();

        // Create args with 5 parameters (exceeds the limit of 4)
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(3))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(4.0))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(5))),
        ];

        let arg_fields = args
            .iter()
            .map(|arg| Field::new("a", arg.data_type(), true).into())
            .collect::<Vec<_>>();

        let func_args = ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 1,
            return_field: Field::new("f", Duration(Microsecond), true).into(),
            config_options: Arc::new(Default::default()),
        };

        let res = udf.invoke_with_args(func_args);

        assert!(
            matches!(res, Err(DataFusionError::Execution(_))),
            "make_dt_interval should return execution error for more than 4 arguments"
        );

        Ok(())
    }
    #[test]
    fn large_day_1_500_000_returns_correct_duration() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(1_500_000)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), 129_600_000_000_000_000);
        Ok(())
    }

    #[test]
    fn boundary_1_491_308_days_and_hours_and_mins() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(1_491_308)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(2)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(8)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), 128_849_018_880_000_000);
        Ok(())
    }

    #[test]
    fn boundary_1_491_309_days_exceeds_i32_mins_safely() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(1_491_309)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), 128_849_097_600_000_000);
        Ok(())
    }

    #[test]
    fn large_negative_day_1_500_000() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(-1_500_000)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), -129_600_000_000_000_000);
        Ok(())
    }

    #[test]
    fn large_hours_exceeding_i32_minutes() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(35_791_395)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), 128_849_022_000_000_000);
        Ok(())
    }

    #[test]
    fn max_representable_duration_microsecond_i64_max() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(106_751_991)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(4)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(54.775_807)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), i64::MAX);
        Ok(())
    }

    #[test]
    fn exceed_i64_max_by_one_microsecond_returns_null() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(106_751_991)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(4)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(54.775_808)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
        Ok(())
    }

    #[test]
    fn overflow_beyond_max_days_returns_null() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(106_751_992)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
        Ok(())
    }

    #[test]
    fn min_representable_duration_microsecond_i64_min() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(-106_751_991)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(-4)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(-54.775_808)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), i64::MIN);
        Ok(())
    }

    #[test]
    fn fractional_second_rounding_with_large_days() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(1_500_000)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.000_000_6)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(!out.is_null(0));
        assert_eq!(out.value(0), 129_600_000_000_000_001);
        Ok(())
    }
    #[test]
    fn underflow_i64_min_by_one_microsecond_returns_null() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(-106_751_991)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(-4)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(-54.775_809)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
        Ok(())
    }

    #[test]
    fn underflow_beyond_min_days_returns_null() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(-106_751_992)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
        Ok(())
    }

    #[test]
    fn extreme_float_seconds_returns_null() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(0), Some(0), Some(0), Some(0)]))
            as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(0), Some(0), Some(0), Some(0)]))
            as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0), Some(0), Some(0), Some(0)]))
            as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![
            Some(-1e20),
            Some(-1e30),
            Some(1e20),
            Some(1e30),
        ])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 4);
        for i in 0..4 {
            assert!(out.is_null(i), "row {i} should be NULL for extreme seconds");
        }
        Ok(())
    }

    #[test]
    fn mixed_sign_intermediate_day_overflow_returns_null() -> Result<()> {
        // (106751992, -24, 0, 0): day * MICROS_PER_DAY overflows i64.
        // In Spark, Math.multiplyExact fails immediately before subtracting 24 hours.
        let days = Arc::new(Int32Array::from(vec![Some(106_751_992)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(-24)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0), "intermediate day overflow must return NULL");
        Ok(())
    }

    #[test]
    fn mixed_sign_intermediate_hour_overflow_returns_null() -> Result<()> {
        // (106751991, 5, -60, 0): days_us + 5 hours overflows i64 before subtracting 60 mins.
        // In Spark, Math.addExact fails at the hour step.
        let days = Arc::new(Int32Array::from(vec![Some(106_751_991)])) as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![Some(5)])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(-60)])) as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![Some(0.0)])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 1);
        assert!(
            out.is_null(0),
            "intermediate hour overflow must return NULL"
        );
        Ok(())
    }

    #[test]
    fn valid_mixed_sign_cancellation_without_overflow() -> Result<()> {
        let days = Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(0), Some(0)]))
            as ArrayRef;
        let hours = Arc::new(Int32Array::from(vec![
            Some(-24),
            Some(-48),
            Some(1),
            Some(0),
        ])) as ArrayRef;
        let mins = Arc::new(Int32Array::from(vec![Some(0), Some(0), Some(-60), Some(1)]))
            as ArrayRef;
        let secs = Arc::new(Float64Array::from(vec![
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(-60.0),
        ])) as ArrayRef;

        let out = run_make_dt_interval(vec![days, hours, mins, secs])?;
        let out = out
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .ok_or_else(|| {
                internal_datafusion_err!("expected DurationMicrosecondArray")
            })?;

        assert_eq!(out.len(), 4);
        for i in 0..4 {
            assert!(!out.is_null(i));
            assert_eq!(out.value(i), 0, "row {i} should be 0 us");
        }
        Ok(())
    }
}
