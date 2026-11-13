// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::{NativeType, logical_date, logical_string};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ExprSchemable, Operator,
    ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignatureClass,
    Volatility, binary_expr,
};
use datafusion_macros::user_doc;

/// Spark-compatible `date_diff` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_diff>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the number of days from `start_date` to `end_date`, as an INT. \
        The result is negative when `end_date` is before `start_date`. \
        DataFusion has no built-in `date_diff`. \
        Strings and timestamps are cast to DATE first, so the time of day is ignored. \
        Returns NULL if either input is NULL.",
    syntax_example = "date_diff(end_date, start_date)",
    alternative_syntax = "datediff(end_date, start_date)",
    sql_example = r#"```sql
> SELECT date_diff(DATE '2009-07-31', DATE '2009-07-30');
1
> SELECT date_diff('2009-07-30', '2009-07-31');
-1
> SELECT date_diff(TIMESTAMP '2009-07-31 23:45:01', TIMESTAMP '2009-07-30 12:34:56');
1
```"#,
    argument(
        name = "end_date",
        description = "DATE, TIMESTAMP or string expression. NULL returns NULL."
    ),
    argument(
        name = "start_date",
        description = "DATE, TIMESTAMP or string expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDateDiff {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkDateDiff {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDateDiff {
    pub fn new() -> Self {
        Self {
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
                        TypeSignatureClass::Native(logical_date()),
                        vec![
                            TypeSignatureClass::Native(logical_string()),
                            TypeSignatureClass::Timestamp,
                        ],
                        NativeType::Date,
                    ),
                ],
                Volatility::Immutable,
            ),
            aliases: vec!["datediff".to_string()],
        }
    }
}

impl ScalarUDFImpl for SparkDateDiff {
    fn name(&self) -> &str {
        "date_diff"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
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

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!(
            "Apache Spark `date_diff` should have been simplified to standard subtraction"
        )
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [end, start] = take_function_args(self.name(), args)?;
        let end = end.cast_to(&DataType::Date32, info.schema())?;
        let start = start.cast_to(&DataType::Date32, info.schema())?;
        Ok(ExprSimplifyResult::Simplified(
            binary_expr(end, Operator::Minus, start)
                .cast_to(&DataType::Int32, info.schema())?,
        ))
    }
}
