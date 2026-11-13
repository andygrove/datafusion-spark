// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use arrow::array::{Array, ArrayRef, BooleanArray};
use arrow::buffer::BooleanBuffer;
use datafusion_common::cast::{
    as_binary_array, as_binary_view_array, as_large_binary_array,
};
use datafusion_common::utils::take_function_args;
use datafusion_functions::utils::make_scalar_function;

use std::sync::Arc;

/// Spark-compatible `is_valid_utf8` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#is_valid_utf8>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns true if the input is valid UTF-8. DataFusion has no built-in `is_valid_utf8`. \
        String input is always valid. Binary input is invalid if it has truncated sequences, stray \
        continuation bytes, overlong encodings, encoded surrogates (U+D800 to U+DFFF) or code points \
        above U+10FFFF. Empty input is valid. Returns NULL if the input is NULL.",
    syntax_example = "is_valid_utf8(str)",
    sql_example = r#"```sql
> SELECT is_valid_utf8('Spark');
true
> SELECT is_valid_utf8(X'C2A9');
true
> SELECT is_valid_utf8(X'61C262');
false
```"#,
    argument(
        name = "str",
        description = "String or binary expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkIsValidUtf8 {
    signature: Signature,
}

impl Default for SparkIsValidUtf8 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkIsValidUtf8 {
    pub fn new() -> Self {
        Self {
            signature: Signature::uniform(
                1,
                vec![
                    DataType::Utf8,
                    DataType::LargeUtf8,
                    DataType::Utf8View,
                    DataType::Binary,
                    DataType::BinaryView,
                    DataType::LargeBinary,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkIsValidUtf8 {
    fn name(&self) -> &str {
        "is_valid_utf8"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(self.name(), DataType::Boolean, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_is_valid_utf8_inner, vec![])(&args.args)
    }
}

fn spark_is_valid_utf8_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array] = take_function_args("is_valid_utf8", args)?;
    match array.data_type() {
        DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8 => {
            Ok(Arc::new(BooleanArray::new(
                BooleanBuffer::new_set(array.len()),
                array.nulls().cloned(),
            )))
        }
        DataType::Binary => Ok(Arc::new(
            as_binary_array(array)?
                .iter()
                .map(|x| x.map(|y| str::from_utf8(y).is_ok()))
                .collect::<BooleanArray>(),
        )),
        DataType::LargeBinary => Ok(Arc::new(
            as_large_binary_array(array)?
                .iter()
                .map(|x| x.map(|y| str::from_utf8(y).is_ok()))
                .collect::<BooleanArray>(),
        )),
        DataType::BinaryView => Ok(Arc::new(
            as_binary_view_array(array)?
                .iter()
                .map(|x| x.map(|y| str::from_utf8(y).is_ok()))
                .collect::<BooleanArray>(),
        )),
        data_type => {
            internal_err!("is_valid_utf8 does not support: {data_type}")
        }
    }
}
