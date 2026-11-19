// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::compute::CastOptions;
use arrow::datatypes::{DataType, TimeUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, plan_err};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility, cast, try_cast,
};
use datafusion_macros::user_doc;

/// Spark-compatible `positive` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#positive>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `expr` unchanged. Accepts integer, floating-point, decimal and interval values. \
        A string is cast to DOUBLE: a malformed string returns NULL with `datafusion.execution.enable_ansi_mode` off \
        and fails with it on. Unlike Spark, a string with a type suffix or a hex float (`'1.5d'`, `'0x1p3'`) is \
        malformed. Returns NULL if the input is NULL.",
    syntax_example = "positive(expr)",
    sql_example = r#"```sql
> SELECT positive(1);
1
> SELECT positive('-1.11');
-1.11
```"#,
    argument(
        name = "expr",
        description = "Numeric, interval or string expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkPositive {
    signature: Signature,
}

impl Default for SparkPositive {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkPositive {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkPositive {
    fn name(&self) -> &str {
        "positive"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [data_type] = take_function_args(self.name(), arg_types)?;
        if data_type.is_string() {
            Ok(DataType::Float64)
        } else {
            Ok(data_type.clone())
        }
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [data_type] = take_function_args(self.name(), arg_types)?;
        // A dictionary is coerced to its value type
        let value_type = match data_type {
            DataType::Dictionary(_, value_type) => value_type.as_ref(),
            other => other,
        };
        match value_type {
            // A NULL literal is treated as DOUBLE, like Spark's implicit cast
            DataType::Null => Ok(vec![DataType::Float64]),
            // The numeric, interval and string types of Spark
            t if t.is_signed_integer()
                || t.is_decimal()
                || t.is_string()
                || matches!(
                    t,
                    DataType::Float32
                        | DataType::Float64
                        | DataType::Interval(_)
                        | DataType::Duration(TimeUnit::Microsecond)
                ) =>
            {
                Ok(vec![t.clone()])
            }
            _ => plan_err!(
                "positive requires a numeric, interval or string input, got {data_type}"
            ),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [arg] = take_function_args(self.name(), args)?;
        if !info.get_data_type(&arg)?.is_string() {
            return Ok(ExprSimplifyResult::Simplified(arg));
        }
        // A malformed string is NULL in legacy mode and an error under ANSI
        let expr = if info.config_options().execution.enable_ansi_mode {
            cast(arg, DataType::Float64)
        } else {
            try_cast(arg, DataType::Float64)
        };
        Ok(ExprSimplifyResult::Simplified(expr))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        let [arg] = take_function_args(self.name(), args.args)?;
        if arg.data_type().is_string() {
            // A malformed string is NULL in legacy mode and an error under ANSI
            let options = CastOptions {
                safe: !enable_ansi_mode,
                ..Default::default()
            };
            arg.cast_to(&DataType::Float64, Some(&options))
        } else {
            Ok(arg)
        }
    }
}
