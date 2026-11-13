// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, plan_err};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::type_coercion::binary::comparison_coercion;
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, Operator, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, Volatility, binary_expr,
};
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::datum::apply_cmp;
use std::sync::Arc;

/// Spark-compatible `equal_null` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#equal_null>
#[user_doc(
    doc_section(label = "Other Functions"),
    description = "Returns true if the two arguments are equal or both NULL, and false otherwise. Never returns NULL. \
        Same as Spark's `<=>` operator and DataFusion's `IS NOT DISTINCT FROM`, which it is rewritten to. \
        The arguments are coerced to a common type, so `equal_null(1::int, 1.0::double)` is true; \
        it is an error if they can't be compared. \
        For floating-point values NaN equals NaN and 0.0 equals -0.0, including inside arrays and structs. \
        DataFusion has no built-in `equal_null`.",
    syntax_example = "equal_null(expr1, expr2)",
    sql_example = r#"```sql
> SELECT equal_null(3, 3);
true
> SELECT equal_null(1, NULL);
false
> SELECT equal_null(NULL, NULL);
true
```"#,
    argument(name = "expr1", description = "Expression of any comparable type."),
    argument(
        name = "expr2",
        description = "Expression of a type comparable with `expr1`."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkEqualNull {
    signature: Signature,
}

impl Default for SparkEqualNull {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkEqualNull {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkEqualNull {
    fn name(&self) -> &str {
        "equal_null"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [lhs, rhs] = arg_types else {
            return plan_err!(
                "Function 'equal_null' expects 2 arguments but received {}",
                arg_types.len()
            );
        };
        // simplify() emits a comparison, and the type coercion pass has already run by then
        let Some(common) = comparison_coercion(lhs, rhs) else {
            return plan_err!(
                "For function 'equal_null' {lhs} and {rhs} are not comparable"
            );
        };
        Ok(vec![common.clone(), common])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, false)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [lhs, rhs] = take_function_args(self.name(), args.args)?;
        apply_cmp(Operator::IsNotDistinctFrom, &lhs, &rhs)
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [lhs, rhs] = take_function_args(self.name(), args)?;
        Ok(ExprSimplifyResult::Simplified(binary_expr(
            lhs,
            Operator::IsNotDistinctFrom,
            rhs,
        )))
    }
}
