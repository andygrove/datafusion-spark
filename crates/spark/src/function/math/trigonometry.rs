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

static CSC_FUNCTION_NAME: &str = "csc";

/// Spark-compatible `csc` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#csc>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the cosecant of `expr` (`1 / sin(expr)`) as a DOUBLE. DataFusion has no \
        built-in `csc`. The argument is cast to DOUBLE. `csc(0)` returns Infinity rather than an error. \
        Returns NULL if the input is NULL, and NaN if it is NaN.",
    syntax_example = "csc(expr)",
    sql_example = r#"```sql
> SELECT csc(1);
1.188395105778121
> SELECT csc(0);
Infinity
```"#,
    argument(
        name = "expr",
        description = "Numeric angle in radians, cast to DOUBLE. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkCsc {
    signature: Signature,
}

impl Default for SparkCsc {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkCsc {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkCsc {
    fn name(&self) -> &str {
        CSC_FUNCTION_NAME
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
        let [arg] = take_function_args(self.name(), &args.args)?;
        spark_csc(arg)
    }
}

fn spark_csc(arg: &ColumnarValue) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Float64(value)) => Ok(ColumnarValue::Scalar(
            ScalarValue::Float64(value.map(|x| 1.0 / x.sin())),
        )),
        ColumnarValue::Array(array) => match array.data_type() {
            DataType::Float64 => Ok(ColumnarValue::Array(Arc::new(
                array
                    .as_primitive::<Float64Type>()
                    .unary::<_, Float64Type>(|x| 1.0 / x.sin()),
            ) as ArrayRef)),
            other => Err(unsupported_data_type_exec_err(
                CSC_FUNCTION_NAME,
                format!("{}", DataType::Float64).as_str(),
                other,
            )),
        },
        other => Err(unsupported_data_type_exec_err(
            CSC_FUNCTION_NAME,
            format!("{}", DataType::Float64).as_str(),
            &other.data_type(),
        )),
    }
}

static SEC_FUNCTION_NAME: &str = "sec";

/// Spark-compatible `sec` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#sec>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the secant of `expr` (`1 / cos(expr)`) as a DOUBLE. DataFusion has no \
        built-in `sec`. The argument is cast to DOUBLE. Returns NULL if the input is NULL, and NaN if it is NaN.",
    syntax_example = "sec(expr)",
    sql_example = r#"```sql
> SELECT sec(0);
1
> SELECT sec(1);
1.850815717680926
```"#,
    argument(
        name = "expr",
        description = "Numeric angle in radians, cast to DOUBLE. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkSec {
    signature: Signature,
}

impl Default for SparkSec {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkSec {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float64], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkSec {
    fn name(&self) -> &str {
        SEC_FUNCTION_NAME
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
        let [arg] = take_function_args(self.name(), &args.args)?;
        spark_sec(arg)
    }
}

fn spark_sec(arg: &ColumnarValue) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Scalar(ScalarValue::Float64(value)) => Ok(ColumnarValue::Scalar(
            ScalarValue::Float64(value.map(|x| 1.0 / x.cos())),
        )),
        ColumnarValue::Array(array) => match array.data_type() {
            DataType::Float64 => Ok(ColumnarValue::Array(Arc::new(
                array
                    .as_primitive::<Float64Type>()
                    .unary::<_, Float64Type>(|x| 1.0 / x.cos()),
            ) as ArrayRef)),
            other => Err(unsupported_data_type_exec_err(
                SEC_FUNCTION_NAME,
                format!("{}", DataType::Float64).as_str(),
                other,
            )),
        },
        other => Err(unsupported_data_type_exec_err(
            SEC_FUNCTION_NAME,
            format!("{}", DataType::Float64).as_str(),
            &other.data_type(),
        )),
    }
}
