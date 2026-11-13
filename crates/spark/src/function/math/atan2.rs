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

/// Spark-compatible `atan2` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#atan2>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the angle in radians, between -pi and pi, between the positive x-axis and \
        the point (`exprX`, `exprY`). Note that the y coordinate comes first. Both arguments are cast to DOUBLE \
        and the result is a DOUBLE. Returns NULL if either input is NULL, and NaN if either input is NaN \
        (and neither is NULL). Infinite inputs and signed zeros follow `java.lang.Math.atan2`: \
        for example `atan2(-0.0, -1.0)` is -pi.",
    syntax_example = "atan2(exprY, exprX)",
    sql_example = r#"```sql
> SELECT atan2(1, 1);
0.785398163397448
> SELECT atan2(0, -1);
3.141592653589793
> SELECT atan2(-0.0::double, -1.0::double);
-3.141592653589793
```"#,
    argument(
        name = "exprY",
        description = "Numeric y coordinate, cast to DOUBLE. NULL returns NULL."
    ),
    argument(
        name = "exprX",
        description = "Numeric x coordinate, cast to DOUBLE. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkAtan2 {
    signature: Signature,
}

impl Default for SparkAtan2 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkAtan2 {
    pub fn new() -> Self {
        Self {
            // Spark only defines atan2 over doubles
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkAtan2 {
    fn name(&self) -> &str {
        "atan2"
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
        make_scalar_function(spark_atan2, vec![])(&args.args)
    }
}

fn spark_atan2(args: &[ArrayRef]) -> Result<ArrayRef> {
    // Spark arg order is atan2(exprY, exprX); Rust computes y.atan2(x).
    let [y, x] = take_function_args("atan2", args)?;
    let y = y.as_primitive::<Float64Type>();
    let x = x.as_primitive::<Float64Type>();
    let result: Float64Array = binary(y, x, |y, x| y.atan2(x))?;
    Ok(Arc::new(result))
}
