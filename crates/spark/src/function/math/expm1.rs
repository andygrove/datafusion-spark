// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::error_utils::unsupported_data_type_exec_err;
use arrow::array::{ArrayRef, AsArray};
use arrow::datatypes::{DataType, Float64Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Spark-compatible `expm1` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#expm1>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `exp(expr) - 1` as a DOUBLE, computed without the precision loss of \
        subtracting 1 from `exp(expr)` when `expr` is close to 0. DataFusion has no built-in `expm1`. \
        Numeric arguments are cast to DOUBLE. Returns NULL if the input is NULL.",
    syntax_example = "expm1(expr)",
    sql_example = r#"```sql
> SELECT expm1(0);
0
> SELECT expm1(1);
1.718281828459045
```"#,
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkExpm1 {
    signature: Signature,
}

impl Default for SparkExpm1 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkExpm1 {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkExpm1 {
    fn name(&self) -> &str {
        "expm1"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        match arg {
            ColumnarValue::Scalar(ScalarValue::Float64(value)) => Ok(
                ColumnarValue::Scalar(ScalarValue::Float64(value.map(|x| x.exp_m1()))),
            ),
            ColumnarValue::Array(array) => match array.data_type() {
                DataType::Float64 => Ok(ColumnarValue::Array(Arc::new(
                    array
                        .as_primitive::<Float64Type>()
                        .unary::<_, Float64Type>(|x| x.exp_m1()),
                )
                    as ArrayRef)),
                other => Err(unsupported_data_type_exec_err(
                    "expm1",
                    format!("{}", DataType::Float64).as_str(),
                    other,
                )),
            },
            other => Err(unsupported_data_type_exec_err(
                "expm1",
                format!("{}", DataType::Float64).as_str(),
                &other.data_type(),
            )),
        }
    }
}
