// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{ArrayRef, LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::{
    as_binary_array, as_binary_view_array, as_large_binary_array,
};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Spark-compatible `make_valid_utf8` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#make_valid_utf8>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the input as a valid UTF-8 string, replacing each invalid byte sequence with the \
        replacement character U+FFFD. DataFusion has no built-in `make_valid_utf8`. String input is \
        already valid and is returned unchanged. Binary input is decoded as UTF-8; overlong encodings and \
        encoded surrogates count as invalid. Returns NULL if the input is NULL.",
    syntax_example = "make_valid_utf8(str)",
    sql_example = r#"```sql
> SELECT make_valid_utf8('Spark');
Spark
> SELECT make_valid_utf8(X'C3A9');
é
> SELECT make_valid_utf8(X'61C262');
a�b
```"#,
    argument(
        name = "str",
        description = "String or binary expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMakeValidUtf8 {
    signature: Signature,
}

impl Default for SparkMakeValidUtf8 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMakeValidUtf8 {
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

impl ScalarUDFImpl for SparkMakeValidUtf8 {
    fn name(&self) -> &str {
        "make_valid_utf8"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [make_valid_utf8] = take_function_args(self.name(), args.arg_fields)?;
        let return_type = match make_valid_utf8.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
                Ok(make_valid_utf8.data_type().clone())
            }
            DataType::Binary | DataType::BinaryView => Ok(DataType::Utf8),
            DataType::LargeBinary => Ok(DataType::LargeUtf8),
            data_type => internal_err!("make_valid_utf8 does not support: {data_type}"),
        }?;
        Ok(Arc::new(Field::new(
            self.name(),
            return_type,
            make_valid_utf8.is_nullable(),
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_make_valid_utf8_inner, vec![])(&args.args)
    }
}

fn spark_make_valid_utf8_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let array = &args[0];
    match &array.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(array.to_owned()),
        DataType::Binary => Ok(Arc::new(
            as_binary_array(&array)?
                .iter()
                .map(|x| x.map(String::from_utf8_lossy))
                .collect::<StringArray>(),
        )),
        DataType::BinaryView => Ok(Arc::new(
            as_binary_view_array(&array)?
                .iter()
                .map(|x| x.map(String::from_utf8_lossy))
                .collect::<StringArray>(),
        )),
        DataType::LargeBinary => Ok(Arc::new(
            as_large_binary_array(&array)?
                .iter()
                .map(|x| x.map(String::from_utf8_lossy))
                .collect::<LargeStringArray>(),
        )),
        data_type => {
            internal_err!("make_valid_utf8 does not support: {data_type}")
        }
    }
}
