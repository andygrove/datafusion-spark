// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::DataType;
use datafusion_common::{Result, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility, binary::try_type_union_resolution, simplify::ExprSimplifyResult, when,
};
use datafusion_macros::user_doc;

/// Spark-compatible `if` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#if>
#[user_doc(
    doc_section(label = "Conditional Functions"),
    description = "Returns `expr1` if `cond` is true, and `expr2` otherwise. A NULL condition counts as false. \
        It is rewritten to `CASE WHEN cond THEN expr1 ELSE expr2 END`, so an error such as division by zero \
        in the branch that isn't taken is not raised. \
        `expr1` and `expr2` are coerced to a common type (`if(true, 10, 20.5)` returns a DOUBLE). \
        It is an error if `cond` is not a boolean or NULL. DataFusion has no built-in `if`.",
    syntax_example = "if(cond, expr1, expr2)",
    sql_example = r#"```sql
> SELECT if(1 < 2, 'a', 'b');
a
> SELECT if(NULL, 'yes', 'no');
no
> SELECT if(true, 1 / 1, 1 / 0);
1
```"#,
    argument(
        name = "cond",
        description = "Boolean expression. NULL is treated as false."
    ),
    argument(
        name = "expr1",
        description = "Expression returned when `cond` is true."
    ),
    argument(
        name = "expr2",
        description = "Expression returned when `cond` is false or NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkIf {
    signature: Signature,
}

impl Default for SparkIf {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkIf {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkIf {
    fn name(&self) -> &str {
        "if"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() != 3 {
            return plan_err!(
                "Function 'if' expects 3 arguments but received {}",
                arg_types.len()
            );
        }

        if arg_types[0] != DataType::Boolean && arg_types[0] != DataType::Null {
            return plan_err!(
                "For function 'if' {} is not a boolean or null",
                arg_types[0]
            );
        }

        let target_types = try_type_union_resolution(&arg_types[1..])?;
        let mut result = vec![DataType::Boolean];
        result.extend(target_types);
        Ok(result)
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(arg_types[1].clone())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("if should have been simplified to case")
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &datafusion_expr::simplify::SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let condition = args[0].clone();
        let then_expr = args[1].clone();
        let else_expr = args[2].clone();

        // Convert IF(condition, then_expr, else_expr) to
        // CASE WHEN condition THEN then_expr ELSE else_expr END
        let case_expr = when(condition, then_expr).otherwise(else_expr)?;

        Ok(ExprSimplifyResult::Simplified(case_expr))
    }
}
