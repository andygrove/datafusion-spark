// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int32Array};
use arrow::datatypes::DataType;
use datafusion_common::types::logical_string;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err};
use datafusion_expr::{
    Coercion, ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use serde::Deserialize;
use serde::de::IgnoredAny;

/// Spark-compatible `json_array_length(jsonArray)`.
/// <https://spark.apache.org/docs/latest/api/sql/index.html#json_array_length>
///
/// Returns the number of elements in the outermost JSON array. The result is NULL
/// for a NULL input, an invalid or truncated JSON string, an empty string, and any
/// valid JSON value that is not an array. Only string arguments are accepted;
/// Spark rejects other types such as integers at analysis time.
///
/// Like Spark, the array is read up to its closing bracket and anything after it
/// is ignored.
///
/// Known differences from Spark, which parses with Jackson configured to accept
/// single-quoted strings and unescaped control characters (Hive compatibility):
/// `serde_json` rejects both, so those inputs give NULL here. `serde_json` also
/// stops at 128 levels of nesting, where Jackson allows 1000.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct JsonArrayLength {
    signature: Signature,
}

impl Default for JsonArrayLength {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonArrayLength {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![Coercion::new_exact(TypeSignatureClass::Native(
                    logical_string(),
                ))],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for JsonArrayLength {
    fn name(&self) -> &str {
        "json_array_length"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(json_array_length_inner, vec![])(&args.args)
    }
}

fn json_array_length_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [array] = take_function_args("json_array_length", args)?;
    let lengths: Int32Array = match array.data_type() {
        DataType::Utf8 => lengths(array.as_string::<i32>().iter()),
        DataType::LargeUtf8 => lengths(array.as_string::<i64>().iter()),
        DataType::Utf8View => lengths(array.as_string_view().iter()),
        DataType::Null => Int32Array::new_null(array.len()),
        other => return exec_err!("json_array_length expects a string, got {other}"),
    };
    Ok(Arc::new(lengths))
}

fn lengths<'a>(strings: impl Iterator<Item = Option<&'a str>>) -> Int32Array {
    strings.map(|s| s.and_then(json_array_length)).collect()
}

/// `None` unless `json` starts with a complete, valid JSON array. Elements are
/// skipped (`IgnoredAny` is zero-sized) and anything after the closing bracket is
/// not read.
fn json_array_length(json: &str) -> Option<i32> {
    let elements =
        Vec::<IgnoredAny>::deserialize(&mut serde_json::Deserializer::from_str(json));
    elements.ok().and_then(|v| i32::try_from(v.len()).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_outermost_elements() {
        assert_eq!(json_array_length("[]"), Some(0));
        assert_eq!(json_array_length("[1,2,3]"), Some(3));
        assert_eq!(json_array_length("[[1,2],[5,6,7]]"), Some(2));
        assert_eq!(
            json_array_length(r#"[1,2,3,[33,44],{"key":[2,3,4]}]"#),
            Some(5)
        );
        assert_eq!(json_array_length(" [1, null, \"a\"] "), Some(3));
    }

    #[test]
    fn non_arrays_and_invalid_json_are_null() {
        for json in ["", "  ", "2", "\"a\"", "null", r#"{"key":"not an array"}"#] {
            assert_eq!(json_array_length(json), None, "{json}");
        }
        for json in ["[1,2,3,4,5", "[1,2", "[1,,2]", "[1 2]", "["] {
            assert_eq!(json_array_length(json), None, "{json}");
        }
    }

    #[test]
    fn content_after_the_closing_bracket_is_ignored() {
        assert_eq!(json_array_length("[1,2] trailing"), Some(2));
    }
}
