// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::logical_string;
use datafusion_common::{Result, ScalarValue, internal_err, plan_err};
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

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!(
            "spark time_trunc should have been simplified to standard date_trunc"
        )
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let fmt_expr = &args[0];

        match fmt_expr.as_literal() {
            Some(ScalarValue::Utf8(Some(unit)))
            | Some(ScalarValue::Utf8View(Some(unit)))
            | Some(ScalarValue::LargeUtf8(Some(unit))) => {
                let valid = ["HOUR", "MINUTE", "SECOND", "MILLISECOND", "MICROSECOND"];
                if !valid.iter().any(|v| v.eq_ignore_ascii_case(unit)) {
                    return plan_err!(
                        "The value of parameter(s) `unit` in `time_trunc` is invalid: expects one of the units 'HOUR', 'MINUTE', 'SECOND', 'MILLISECOND', 'MICROSECOND', but got '{unit}'. SQLSTATE: 22023"
                    );
                }
            }
            // A NULL unit returns NULL, like Spark
            Some(value) if value.is_null() => {
                let data_type = info.get_data_type(&args[1])?;
                return Ok(ExprSimplifyResult::Simplified(Expr::Literal(
                    ScalarValue::try_from(&data_type)?,
                    None,
                )));
            }
            _ => {
                return plan_err!(
                    "First argument of `TIME_TRUNC` must be non-null scalar Utf8"
                );
            }
        }

        Ok(ExprSimplifyResult::Simplified(Expr::ScalarFunction(
            ScalarFunction::new_udf(datafusion_functions::datetime::date_trunc(), args),
        )))
    }
}
