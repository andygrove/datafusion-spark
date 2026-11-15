// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::{str::FromStr, sync::Arc};

use arrow::compute::DatePart;
use arrow::datatypes::{DataType, Field, FieldRef, TimeUnit};
use datafusion_common::types::{NativeType, logical_date, logical_string};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err, plan_err};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ExprSchemable, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `trunc` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#trunc>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns `date` truncated to the unit given by `fmt`, as a date. \
        Unlike DataFusion's `date_trunc`, it takes the date first and the unit second, only accepts dates \
        (a string is cast to a date), and also accepts Spark's unit aliases `yyyy` and `yy` for year and `mon` and `mm` for month. \
        Returns NULL if `date` is NULL. `fmt` must be a constant string: a NULL or unsupported `fmt` is a planning error, \
        where Spark returns NULL.",
    syntax_example = "trunc(date, fmt)",
    sql_example = r#"```sql
> SELECT trunc('2009-02-12'::date, 'YEAR');
2009-01-01
> SELECT trunc('2009-02-12'::date, 'MM');
2009-02-01
> SELECT trunc('2009-02-12'::date, 'WEEK');
2009-02-09
```"#,
    argument(
        name = "date",
        description = "Date expression to truncate. A string is cast to a date. NULL returns NULL."
    ),
    argument(
        name = "fmt",
        description = "Constant string unit, case-insensitive: `year` (`yyyy`, `yy`), `quarter`, `month` (`mon`, `mm`), `week` (truncates to Monday) or `day`, or another spelling of these that DataFusion accepts (such as `months`)."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTrunc {
    signature: Signature,
}

impl Default for SparkTrunc {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTrunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_date()),
                        vec![TypeSignatureClass::Native(logical_string())],
                        NativeType::Date,
                    ),
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkTrunc {
    fn name(&self) -> &str {
        "trunc"
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
            args.arg_fields[0].data_type().clone(),
            nullable,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("spark trunc should have been simplified to standard date_trunc")
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [dt_expr, fmt_expr] = take_function_args(self.name(), args)?;

        let fmt = match fmt_expr.as_literal() {
            Some(ScalarValue::Utf8(Some(v)))
            | Some(ScalarValue::Utf8View(Some(v)))
            | Some(ScalarValue::LargeUtf8(Some(v))) => v.to_lowercase(),
            _ => {
                return plan_err!(
                    "Second argument of `TRUNC` must be non-null scalar Utf8"
                );
            }
        };

        // Map Spark-specific fmt aliases to datafusion ones
        let fmt = match fmt.as_str() {
            "yy" | "yyyy" => "year",
            "mm" | "mon" => "month",
            other => other,
        };

        // Accept shared DatePart aliases for the date parts supported by TRUNC.
        if !matches!(
            DatePart::from_str(fmt),
            Ok(DatePart::Year
                | DatePart::Month
                | DatePart::Day
                | DatePart::Week
                | DatePart::Quarter)
        ) {
            return plan_err!(
                "The format argument of `TRUNC` must represent a year, month, day, week, or quarter."
            );
        }
        let return_type = dt_expr.get_type(info.schema())?;

        let fmt_expr = Expr::Literal(ScalarValue::new_utf8(fmt), None);

        // Spark uses Dates so we need to cast to timestamp and back to work with datafusion's date_trunc
        Ok(ExprSimplifyResult::Simplified(
            Expr::ScalarFunction(ScalarFunction::new_udf(
                datafusion_functions::datetime::date_trunc(),
                vec![
                    fmt_expr,
                    dt_expr.cast_to(
                        &DataType::Timestamp(TimeUnit::Nanosecond, None),
                        info.schema(),
                    )?,
                ],
            ))
            .cast_to(&return_type, info.schema())?,
        ))
    }
}
