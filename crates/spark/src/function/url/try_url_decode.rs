// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::ArrayRef;
use arrow::datatypes::DataType;

use datafusion_common::Result;
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use crate::function::url::url_decode::{
    OnDecodeError, UrlDecode, spark_handled_url_decode,
};

/// Spark-compatible `try_url_decode` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#try_url_decode>
#[user_doc(
    doc_section(label = "URL Functions"),
    description = "Same as `url_decode`, except that a value which cannot be decoded returns NULL for that row \
        instead of an error. DataFusion has no built-in `try_url_decode`.",
    syntax_example = "try_url_decode(str)",
    sql_example = r#"```sql
> SELECT try_url_decode('Spark+SQL%21');
Spark SQL!
> SELECT try_url_decode('https%3%2F%2Fspark.apache.org');
NULL
```"#,
    argument(
        name = "str",
        description = "String expression to decode. NULL, or a value that cannot be decoded, returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TryUrlDecode {
    signature: Signature,
    url_decoder: UrlDecode,
}

impl Default for TryUrlDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl TryUrlDecode {
    pub fn new() -> Self {
        Self {
            signature: Signature::string(1, Volatility::Immutable),
            url_decoder: UrlDecode::new(),
        }
    }
}

impl ScalarUDFImpl for TryUrlDecode {
    fn name(&self) -> &str {
        "try_url_decode"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        self.url_decoder.return_type(arg_types)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs { args, .. } = args;
        make_scalar_function(spark_try_url_decode, vec![])(&args)
    }
}

fn spark_try_url_decode(args: &[ArrayRef]) -> Result<ArrayRef> {
    spark_handled_url_decode(args, OnDecodeError::Null)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::StringArray;
    use datafusion_common::cast::as_string_array;

    use super::*;

    #[test]
    fn test_try_decode_error_handled() -> Result<()> {
        let input = Arc::new(StringArray::from(vec![
            Some("http%3A%2F%2spark.apache.org"), // '%2s' is not a valid percent encoded character
            // Valid cases
            Some("https%3A%2F%2Fspark.apache.org"),
            None,
        ]));

        let expected =
            StringArray::from(vec![None, Some("https://spark.apache.org"), None]);

        let result = spark_try_url_decode(&[input as ArrayRef])?;
        let result = as_string_array(&result)?;

        assert_eq!(&expected, result);
        Ok(())
    }
}
