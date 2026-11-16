// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::logical_string;
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, internal_err, plan_datafusion_err,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `time_trunc` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#time_trunc>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the time `time` truncated to the unit given by `unit`, with the same type as `time`. \
        DataFusion has no `time_trunc`; this rewrites to DataFusion's `date_trunc`, but only accepts time input and the units \
        `HOUR`, `MINUTE`, `SECOND`, `MILLISECOND` and `MICROSECOND` (case-insensitive). Any other unit is an error, \
        with the same message and SQLSTATE 22023 as Spark. Returns NULL if `unit` or `time` is NULL.",
    syntax_example = "time_trunc(unit, time)",
    sql_example = r#"```sql
> SELECT time_trunc('HOUR', '09:32:05.123456'::time);
09:00:00
> SELECT time_trunc('MILLISECOND', '09:32:05.123456'::time);
09:32:05.123
> SELECT time_trunc(NULL, '09:32:05.123456'::time);
NULL
```"#,
    argument(
        name = "unit",
        description = "Constant string unit: `HOUR`, `MINUTE`, `SECOND`, `MILLISECOND` or `MICROSECOND`, case-insensitive. NULL returns NULL."
    ),
    argument(
        name = "time",
        description = "Time expression to truncate. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTimeTrunc {
    signature: Signature,
}

impl Default for SparkTimeTrunc {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTimeTrunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                    Coercion::new_exact(TypeSignatureClass::Time),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkTimeTrunc {
    fn name(&self) -> &str {
        "time_trunc"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());

        Ok(Arc::new(Field::new(
            self.name(),
            args.arg_fields[1].data_type().clone(),
            nullable,
        )))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let unit = match &args.args[0] {
            ColumnarValue::Scalar(unit) => Some(unit),
            ColumnarValue::Array(_) => None,
        };
        match check_unit(unit).map_err(|e| exec_datafusion_err!("{e}"))? {
            Unit::Valid => {
                datafusion_functions::datetime::date_trunc().invoke_with_args(args)
            }
            // A NULL unit returns NULL, like Spark
            Unit::Null => {
                let data_type = args.return_field.data_type();
                Ok(ColumnarValue::Scalar(ScalarValue::try_from(data_type)?))
            }
        }
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        match check_unit(args[0].as_literal()).map_err(|e| plan_datafusion_err!("{e}"))? {
            Unit::Valid => Ok(ExprSimplifyResult::Simplified(Expr::ScalarFunction(
                ScalarFunction::new_udf(
                    datafusion_functions::datetime::date_trunc(),
                    args,
                ),
            ))),
            // A NULL unit returns NULL, like Spark
            Unit::Null => {
                let data_type = info.get_data_type(&args[1])?;
                Ok(ExprSimplifyResult::Simplified(Expr::Literal(
                    ScalarValue::try_from(&data_type)?,
                    None,
                )))
            }
        }
    }
}

/// A checked `unit` argument
enum Unit {
    Valid,
    Null,
}

/// Checks the `unit` argument. Returns the error message for an invalid or
/// non-constant unit, so the caller can pick the error kind.
fn check_unit(unit: Option<&ScalarValue>) -> std::result::Result<Unit, String> {
    match unit {
        Some(ScalarValue::Utf8(Some(unit)))
        | Some(ScalarValue::Utf8View(Some(unit)))
        | Some(ScalarValue::LargeUtf8(Some(unit))) => {
            let valid = ["HOUR", "MINUTE", "SECOND", "MILLISECOND", "MICROSECOND"];
            if valid.iter().any(|v| v.eq_ignore_ascii_case(unit)) {
                Ok(Unit::Valid)
            } else {
                Err(format!(
                    "The value of parameter(s) `unit` in `time_trunc` is invalid: expects one of the units 'HOUR', 'MINUTE', 'SECOND', 'MILLISECOND', 'MICROSECOND', but got '{unit}'. SQLSTATE: 22023"
                ))
            }
        }
        Some(value) if value.is_null() => Ok(Unit::Null),
        _ => Err("First argument of `TIME_TRUNC` must be a constant string".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Array, ArrayRef, Time64NanosecondArray};
    use datafusion_common::config::ConfigOptions;

    /// `HH:MM:SS.ffffff` as nanoseconds since midnight
    fn nanos(hours: i64, minutes: i64, seconds: i64, micros: i64) -> i64 {
        ((hours * 60 + minutes) * 60 + seconds) * 1_000_000_000 + micros * 1_000
    }

    /// Calls `invoke_with_args` directly, without `simplify`
    fn invoke(unit: ScalarValue, time: ColumnarValue) -> Result<ColumnarValue> {
        let time_type = time.data_type();
        let number_rows = match &time {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        SparkTimeTrunc::new().invoke_with_args(ScalarFunctionArgs {
            arg_fields: vec![
                Arc::new(Field::new("unit", unit.data_type(), true)),
                Arc::new(Field::new("time", time_type.clone(), true)),
            ],
            args: vec![ColumnarValue::Scalar(unit), time],
            number_rows,
            return_field: Arc::new(Field::new("time_trunc", time_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn scalar_time(value: i64) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Time64Nanosecond(Some(value)))
    }

    fn unit(value: &str) -> ScalarValue {
        ScalarValue::Utf8(Some(value.to_string()))
    }

    #[test]
    fn invoke_truncates_a_scalar_to_every_unit() {
        let time = nanos(9, 32, 5, 123_456);
        for (name, expected) in [
            ("HOUR", nanos(9, 0, 0, 0)),
            ("MINUTE", nanos(9, 32, 0, 0)),
            ("SECOND", nanos(9, 32, 5, 0)),
            ("MILLISECOND", nanos(9, 32, 5, 123_000)),
            ("MICROSECOND", nanos(9, 32, 5, 123_456)),
            ("hour", nanos(9, 0, 0, 0)),
        ] {
            let ColumnarValue::Scalar(result) =
                invoke(unit(name), scalar_time(time)).unwrap()
            else {
                panic!("expected a scalar for {name}");
            };
            assert_eq!(
                result,
                ScalarValue::Time64Nanosecond(Some(expected)),
                "{name}"
            );
        }
    }

    #[test]
    fn invoke_truncates_an_array() {
        let times: ArrayRef = Arc::new(Time64NanosecondArray::from(vec![
            Some(nanos(9, 32, 5, 123_456)),
            Some(nanos(22, 45, 30, 654_321)),
            None,
        ]));
        for unit in [
            ScalarValue::Utf8(Some("minute".to_string())),
            ScalarValue::LargeUtf8(Some("Minute".to_string())),
            ScalarValue::Utf8View(Some("MINUTE".to_string())),
        ] {
            let result = invoke(unit, ColumnarValue::Array(Arc::clone(&times)))
                .unwrap()
                .into_array(3)
                .unwrap();
            let expected: ArrayRef = Arc::new(Time64NanosecondArray::from(vec![
                Some(nanos(9, 32, 0, 0)),
                Some(nanos(22, 45, 0, 0)),
                None,
            ]));
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn invoke_rejects_an_invalid_unit() {
        // DAY is valid for date_trunc but not for time_trunc
        for name in ["MINS", "DAY"] {
            let err = invoke(unit(name), scalar_time(0)).unwrap_err();
            assert!(err.to_string().starts_with("Execution error:"), "{err}");
            assert!(
                err.to_string().contains(&format!(
                    "expects one of the units 'HOUR', 'MINUTE', 'SECOND', 'MILLISECOND', 'MICROSECOND', but got '{name}'. SQLSTATE: 22023"
                )),
                "{err}"
            );
        }
    }

    #[test]
    fn invoke_returns_null_for_a_null_unit() {
        let result = invoke(ScalarValue::Utf8(None), scalar_time(nanos(9, 32, 5, 0)))
            .unwrap()
            .into_array(1)
            .unwrap();
        assert_eq!(
            result.data_type(),
            &DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond)
        );
        assert!(result.is_null(0));
    }

    #[test]
    fn invoke_truncates_every_time_type() {
        // 09:32:05.123456 truncated to the minute is 09:32:00
        for (time, expected) in [
            (
                ScalarValue::Time32Second(Some(34_325)),
                ScalarValue::Time32Second(Some(34_320)),
            ),
            (
                ScalarValue::Time32Millisecond(Some(34_325_123)),
                ScalarValue::Time32Millisecond(Some(34_320_000)),
            ),
            (
                ScalarValue::Time64Microsecond(Some(34_325_123_456)),
                ScalarValue::Time64Microsecond(Some(34_320_000_000)),
            ),
        ] {
            let ColumnarValue::Scalar(result) =
                invoke(unit("MINUTE"), ColumnarValue::Scalar(time)).unwrap()
            else {
                panic!("expected a scalar");
            };
            assert_eq!(result, expected);
        }
    }
}
