// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Field};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions_nested::repeat::ArrayRepeat;
use datafusion_macros::user_doc;
use std::sync::Arc;

use crate::function::null_utils::{
    NullMaskResolution, apply_null_mask, compute_null_mask,
};

/// Spark-compatible `array_repeat` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_repeat>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns an array that contains `element` `count` times. \
        Unlike DataFusion's `array_repeat`, a NULL `count` returns NULL. \
        A NULL `element` is repeated like any other value, and a `count` of 0 or less returns an empty array.",
    syntax_example = "array_repeat(element, count)",
    sql_example = r#"```sql
> SELECT array_repeat('123', 2);
[123, 123]
> SELECT array_repeat(NULL, 2);
[NULL, NULL]
> SELECT array_repeat('123', NULL);
NULL
```"#,
    argument(
        name = "element",
        description = "Value to repeat, of any type. NULL is repeated."
    ),
    argument(
        name = "count",
        description = "Integer number of times to repeat `element`. Non-integer types are an error. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayRepeat {
    signature: Signature,
}

impl Default for SparkArrayRepeat {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayRepeat {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayRepeat {
    fn name(&self) -> &str {
        "array_repeat"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new_list_field(
            arg_types[0].clone(),
            true,
        ))))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        spark_array_repeat(args)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [first_type, second_type] = take_function_args(self.name(), arg_types)?;

        // Coerce the second argument to Int64/UInt64 if it's a numeric type
        let second = match second_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Null => DataType::Int64,
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                DataType::UInt64
            }
            _ => return exec_err!("count must be an integer type"),
        };

        Ok(vec![first_type.clone(), second])
    }
}

/// This is a Spark-specific wrapper around DataFusion's array_repeat that returns NULL
/// if the count argument is NULL (Spark behavior), whereas DataFusion's array_repeat ignores NULLs.
fn spark_array_repeat(args: ScalarFunctionArgs) -> Result<ColumnarValue> {
    let ScalarFunctionArgs {
        args: arg_values,
        arg_fields,
        number_rows,
        return_field,
        config_options,
    } = args;
    let return_type = return_field.data_type().clone();

    // A NULL element should be repeated into the array, not cause a NULL result.
    let null_mask = compute_null_mask(&arg_values[1..]);

    // If count is null then return NULL immediately
    if matches!(null_mask, NullMaskResolution::ReturnNull) {
        return Ok(ColumnarValue::Scalar(ScalarValue::try_from(return_type)?));
    }

    let array_repeat_func = ArrayRepeat::new();
    let func_args = ScalarFunctionArgs {
        args: arg_values,
        arg_fields,
        number_rows,
        return_field,
        config_options,
    };
    let result = array_repeat_func.invoke_with_args(func_args)?;

    apply_null_mask(result, null_mask, &return_type)
}
