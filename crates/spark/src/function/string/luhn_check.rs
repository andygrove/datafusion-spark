// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, AsArray, BooleanArray};
use arrow::datatypes::DataType;
use arrow::datatypes::DataType::Boolean;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `luhn_check` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#luhn_check>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns true if the string is a number that passes the Luhn checksum, as used for credit \
        card numbers. DataFusion has no built-in `luhn_check`. The string must contain only the digits 0-9: \
        spaces, letters or any other character make the result false, and so does an empty string. \
        Numeric input is cast to a string first. Returns NULL if the input is NULL.",
    syntax_example = "luhn_check(str)",
    sql_example = r#"```sql
> SELECT luhn_check('79927398713');
true
> SELECT luhn_check('79927398714');
false
> SELECT luhn_check('4111111 111111111');
false
```"#,
    argument(
        name = "str",
        description = "String of digits to check. Numeric input is cast to a string. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLuhnCheck {
    signature: Signature,
}

impl Default for SparkLuhnCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkLuhnCheck {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Utf8]),
                    TypeSignature::Exact(vec![DataType::Utf8View]),
                    TypeSignature::Exact(vec![DataType::LargeUtf8]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkLuhnCheck {
    fn name(&self) -> &str {
        "luhn_check"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(Boolean)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [array] = take_function_args(self.name(), &args.args)?;

        match array {
            ColumnarValue::Array(array) => match array.data_type() {
                DataType::Utf8View => {
                    let str_array = array.as_string_view();
                    let values = str_array
                        .iter()
                        .map(|s| s.map(luhn_check_impl))
                        .collect::<BooleanArray>();
                    Ok(ColumnarValue::Array(Arc::new(values)))
                }
                DataType::Utf8 => {
                    let str_array = array.as_string::<i32>();
                    let values = str_array
                        .iter()
                        .map(|s| s.map(luhn_check_impl))
                        .collect::<BooleanArray>();
                    Ok(ColumnarValue::Array(Arc::new(values)))
                }
                DataType::LargeUtf8 => {
                    let str_array = array.as_string::<i64>();
                    let values = str_array
                        .iter()
                        .map(|s| s.map(luhn_check_impl))
                        .collect::<BooleanArray>();
                    Ok(ColumnarValue::Array(Arc::new(values)))
                }
                other => {
                    exec_err!("Unsupported data type {other:?} for function `luhn_check`")
                }
            },
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some(s)))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(Some(s))) => Ok(
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(luhn_check_impl(s)))),
            ),
            ColumnarValue::Scalar(ScalarValue::Utf8(None))
            | ColumnarValue::Scalar(ScalarValue::LargeUtf8(None))
            | ColumnarValue::Scalar(ScalarValue::Utf8View(None)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)))
            }
            other => {
                exec_err!("Unsupported data type {other:?} for function `luhn_check`")
            }
        }
    }
}

/// Validates a string using the Luhn algorithm.
/// Returns `true` if the input is a valid Luhn number.
fn luhn_check_impl(input: &str) -> bool {
    let mut sum = 0u32;
    let mut alt = false;
    let mut digits_processed = 0;

    for b in input.as_bytes().iter().rev() {
        let digit = match b {
            b'0'..=b'9' => {
                digits_processed += 1;
                b - b'0'
            }
            _ => return false,
        };

        let mut val = digit as u32;
        if alt {
            val *= 2;
            if val > 9 {
                val -= 9;
            }
        }
        sum += val;
        alt = !alt;
    }

    digits_processed > 0 && sum.is_multiple_of(10)
}
