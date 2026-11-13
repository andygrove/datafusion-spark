// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Float64Array};
use arrow::compute::kernels::arity::binary;
use arrow::datatypes::{DataType, Float64Type};
use datafusion_common::Result;
use datafusion_common::utils::take_function_args;
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark-compatible `hypot` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#hypot>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `sqrt(expr1^2 + expr2^2)` as a DOUBLE, computed without intermediate \
        overflow or underflow, like `java.lang.Math.hypot`. DataFusion has no built-in `hypot`. \
        Both arguments are cast to DOUBLE. Returns NULL if either input is NULL. If either input is \
        infinite the result is Infinity, even if the other is NaN. Otherwise a NaN input returns NaN.",
    syntax_example = "hypot(expr1, expr2)",
    sql_example = r#"```sql
> SELECT hypot(3, 4);
5
> SELECT hypot('Infinity'::double, 'NaN'::double);
Infinity
> SELECT hypot(NULL::double, 4.0::double);
NULL
```"#,
    argument(
        name = "expr1",
        description = "Numeric expression, cast to DOUBLE. NULL returns NULL."
    ),
    argument(
        name = "expr2",
        description = "Numeric expression, cast to DOUBLE. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkHypot {
    signature: Signature,
}

impl Default for SparkHypot {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkHypot {
    pub fn new() -> Self {
        Self {
            // Spark only defines hypot over doubles
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkHypot {
    fn name(&self) -> &str {
        "hypot"
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
        make_scalar_function(spark_hypot, vec![])(&args.args)
    }
}

fn spark_hypot(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [x, y] = take_function_args("hypot", args)?;

    let x = x.as_primitive::<Float64Type>();
    let y = y.as_primitive::<Float64Type>();
    let result: Float64Array = binary(x, y, |a, b| a.hypot(b))?;
    Ok(Arc::new(result))
}
