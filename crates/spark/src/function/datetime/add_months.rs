// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::ops::Add;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, IntervalUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ExprSchemable, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `add_months` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#add_months>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the date that is `num_months` months after `start_date`. \
        A negative `num_months` moves the date backwards. DataFusion has no built-in `add_months`. \
        If the day of the month does not exist in the result month, the last day of that month is returned \
        (so subtracting 5 months from 2016-07-30 gives 2016-02-29). \
        Returns NULL if either input is NULL. A result outside the supported date range returns an error. \
        `num_months` must be an INT. Integer literals are BIGINT in DataFusion, so cast them (for example `1::INT`). \
        `start_date` must be a DATE; strings are not implicitly cast.",
    syntax_example = "add_months(start_date, num_months)",
    sql_example = r#"```sql
> SELECT add_months(DATE '2016-07-30', 1::INT);
2016-08-30
> SELECT add_months(DATE '2016-07-30', -5::INT);
2016-02-29
> SELECT add_months(DATE '2016-07-30', NULL::INT);
NULL
```"#,
    argument(
        name = "start_date",
        description = "DATE expression. NULL returns NULL."
    ),
    argument(
        name = "num_months",
        description = "INT number of months to add. Negative values subtract months. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkAddMonths {
    signature: Signature,
}

impl Default for SparkAddMonths {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkAddMonths {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Date32, DataType::Int32],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkAddMonths {
    fn name(&self) -> &str {
        "add_months"
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
            DataType::Date32,
            nullable,
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [date_arg, months_arg] = take_function_args("add_months", args)?;
        let interval = months_arg
            .cast_to(&DataType::Interval(IntervalUnit::YearMonth), info.schema())?;
        Ok(ExprSimplifyResult::Simplified(date_arg.add(interval)))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("invoke should not be called on a simplified add_months() function")
    }
}
