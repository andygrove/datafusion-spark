// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, new_null_array};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::SingleRowListArrayBuilder;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_functions_nested::make_array::{array_array, coerce_types_inner};
use datafusion_macros::user_doc;

use crate::function::functions_nested_utils::make_scalar_function;

const ARRAY_FIELD_DEFAULT_NAME: &str = "element";

/// Spark-compatible `array` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns an array of the given elements, coerced to a common type. NULL elements are kept, \
        and with no arguments the result is an empty array. \
        Unlike DataFusion's `make_array`, the list's element field is named `element`, as in Spark, rather than `item`.",
    syntax_example = "array(expr, ...)",
    sql_example = r#"```sql
> SELECT array(1, 2, 3);
[1, 2, 3]
> SELECT array(1, NULL, 3);
[1, NULL, 3]
> SELECT array();
[]
```"#,
    argument(
        name = "expr",
        description = "Zero or more elements of any type that can be coerced to a common type."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArray {
    signature: Signature,
}

impl Default for SparkArray {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArray {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArray {
    fn name(&self) -> &str {
        "array"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let data_types = args
            .arg_fields
            .iter()
            .map(|f| f.data_type())
            .cloned()
            .collect::<Vec<_>>();

        let mut expr_type = DataType::Null;
        for arg_type in &data_types {
            if !arg_type.equals_datatype(&DataType::Null) {
                expr_type = arg_type.clone();
                break;
            }
        }

        let return_type = DataType::List(Arc::new(Field::new(
            ARRAY_FIELD_DEFAULT_NAME,
            expr_type,
            true,
        )));

        Ok(Arc::new(Field::new(
            "this_field_name_is_irrelevant",
            return_type,
            false,
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        make_scalar_function(make_array_inner)(args.as_slice())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.is_empty() {
            Ok(vec![])
        } else {
            coerce_types_inner(arg_types, self.name())
        }
    }
}

/// `make_array_inner` is the implementation of the `make_array` function.
/// Constructs an array using the input `data` as `ArrayRef`.
/// Returns a reference-counted `Array` instance result.
pub fn make_array_inner(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let mut data_type = DataType::Null;
    for arg in arrays {
        let arg_data_type = arg.data_type();
        if !arg_data_type.equals_datatype(&DataType::Null) {
            data_type = arg_data_type.clone();
            break;
        }
    }

    match data_type {
        // Either an empty array or all nulls:
        DataType::Null => {
            let length = arrays.iter().map(|a| a.len()).sum();
            // By default Int32
            let array = new_null_array(&DataType::Null, length);
            Ok(Arc::new(
                SingleRowListArrayBuilder::new(array)
                    .with_nullable(true)
                    .with_field_name(Some(ARRAY_FIELD_DEFAULT_NAME.to_string()))
                    .build_list_array(),
            ))
        }
        _ => array_array::<i32>(arrays, data_type, ARRAY_FIELD_DEFAULT_NAME),
    }
}
