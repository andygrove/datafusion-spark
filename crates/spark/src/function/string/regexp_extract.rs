// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Int32Array, StringArray, StringBuilder, new_null_array,
};
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::{Result, ScalarValue, exec_datafusion_err, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use regex::{CaptureLocations, Regex};

use crate::function::error_utils::invalid_arg_count_exec_err;

const NAME: &str = "regexp_extract";

/// Spark-compatible `regexp_extract` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
#[user_doc(
    doc_section(label = "Regular Expression Functions"),
    description = "Returns the text matched by group `idx` of the first match of `regexp` in `str`. \
        Unlike DataFusion's `regexp_match`, which returns a list of every capture group, this returns a single string. \
        Returns an empty string if there is no match, or if the group did not take part in the match. \
        `idx` is only checked when the pattern matches: an `idx` outside `0..=groupCount` is an error \
        (`INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX`) on a row that matches, and an empty string on a row that does not. \
        An invalid pattern is an error (`INVALID_PARAMETER_VALUE.PATTERN`). Returns NULL if any argument is NULL. \
        Patterns use Rust `regex` syntax, not Java's `java.util.regex` as in Spark: Java-only features such as \
        lookaround and backreferences are rejected as invalid patterns, and `\\d` matches any Unicode digit, not just ASCII 0-9.",
    syntax_example = "regexp_extract(str, regexp[, idx])",
    sql_example = r#"```sql
> SELECT regexp_extract('1a 2b 14m', '(\d+)([a-z]+)');
1
> SELECT regexp_extract('1a 2b 14m', '(\d+)([a-z]+)', 0);
1a
> SELECT regexp_extract('1a 2b 14m', '(\d+)([a-z]+)', 2);
a
> SELECT regexp_extract('abc', '(\d+)', 1);
-- empty string
> SELECT regexp_extract('a1', '(\d+)', NULL);
NULL
```"#,
    argument(
        name = "str",
        description = "String expression to search. NULL returns NULL."
    ),
    argument(
        name = "regexp",
        description = "String regular expression in Rust `regex` syntax. NULL returns NULL."
    ),
    argument(
        name = "idx",
        description = "Optional integer capture group index. Defaults to 1; 0 is the whole match. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkRegexpExtract {
    signature: Signature,
}

impl Default for SparkRegexpExtract {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkRegexpExtract {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkRegexpExtract {
    fn name(&self) -> &str {
        NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if !(2..=3).contains(&arg_types.len()) {
            return Err(invalid_arg_count_exec_err(NAME, (2, 3), arg_types.len()));
        }
        let is_text = |t: &DataType| {
            matches!(
                t,
                DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Null
            )
        };
        if !is_text(&arg_types[0]) || !is_text(&arg_types[1]) {
            return exec_err!(
                "{NAME} expects string str and regexp, got {} and {}",
                arg_types[0],
                arg_types[1]
            );
        }
        let mut coerced = vec![DataType::Utf8, DataType::Utf8];
        if let Some(idx) = arg_types.get(2) {
            if !idx.is_integer() && *idx != DataType::Null {
                return exec_err!("{NAME} expects an INT idx, got {idx}");
            }
            coerced.push(DataType::Int32);
        }
        Ok(coerced)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let all_scalar = args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));
        let len = if all_scalar { 1 } else { args.number_rows };

        // The usual call has a literal pattern and idx: keep those as scalars so
        // the pattern is compiled once and no per-row copies are allocated.
        let pattern = match &args.args[1] {
            ColumnarValue::Scalar(scalar) => match scalar_str(scalar)? {
                Some(pattern) => Pattern::Scalar(pattern),
                None => return null_result(len, all_scalar),
            },
            ColumnarValue::Array(array) => Pattern::Array(array.as_string::<i32>()),
        };
        let index = match args.args.get(2) {
            None => Index::Scalar(1),
            Some(ColumnarValue::Scalar(ScalarValue::Int32(Some(idx)))) => {
                Index::Scalar(*idx)
            }
            Some(ColumnarValue::Scalar(ScalarValue::Int32(None) | ScalarValue::Null)) => {
                return null_result(len, all_scalar);
            }
            Some(ColumnarValue::Scalar(other)) => {
                return exec_err!("{NAME} expects an INT idx, got {}", other.data_type());
            }
            Some(ColumnarValue::Array(array)) => Index::Array(as_int32_array(array)?),
        };

        let strings = args.args[0].to_array(len)?;
        let result = extract(strings.as_string::<i32>(), pattern, index)?;
        if all_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

#[derive(Clone, Copy)]
enum Pattern<'a> {
    Scalar(&'a str),
    Array(&'a StringArray),
}

#[derive(Clone, Copy)]
enum Index<'a> {
    Scalar(i32),
    Array(&'a Int32Array),
}

fn scalar_str(scalar: &ScalarValue) -> Result<Option<&str>> {
    match scalar {
        ScalarValue::Null => Ok(None),
        other => other.try_as_str().ok_or_else(|| {
            exec_datafusion_err!(
                "{NAME} expects a string regexp, got {}",
                other.data_type()
            )
        }),
    }
}

fn null_result(len: usize, all_scalar: bool) -> Result<ColumnarValue> {
    Ok(if all_scalar {
        ColumnarValue::Scalar(ScalarValue::Utf8(None))
    } else {
        ColumnarValue::Array(new_null_array(&DataType::Utf8, len))
    })
}

fn extract(strings: &StringArray, pattern: Pattern, index: Index) -> Result<ArrayRef> {
    // Each result is a substring of its input, so the input bytes bound the output.
    let mut builder =
        StringBuilder::with_capacity(strings.len(), strings.value_data().len());
    // A scalar pattern is compiled on the first non-NULL row and reused. A column
    // pattern keeps only the last one compiled, since it rarely changes per row.
    // The capture buffer is reused across rows instead of allocated per match.
    let mut cached: Option<(&str, Regex, CaptureLocations)> = None;

    let scalar_pattern = matches!(pattern, Pattern::Scalar(_));

    for row in 0..strings.len() {
        let row_pattern = match pattern {
            Pattern::Scalar(pattern) => Some(pattern),
            Pattern::Array(patterns) => {
                (!patterns.is_null(row)).then(|| patterns.value(row))
            }
        };
        let idx = match index {
            Index::Scalar(idx) => Some(idx),
            Index::Array(indexes) => (!indexes.is_null(row)).then(|| indexes.value(row)),
        };
        let (false, Some(pattern), Some(idx)) = (strings.is_null(row), row_pattern, idx)
        else {
            builder.append_null();
            continue;
        };
        let (_, regex, locations) = match &mut cached {
            Some(entry) if scalar_pattern || entry.0 == pattern => entry,
            _ => {
                let regex = compile(pattern)?;
                let locations = regex.capture_locations();
                cached.insert((pattern, regex, locations))
            }
        };
        let subject = strings.value(row);
        // Like Spark, idx is only checked when the pattern matches.
        let text = match regex.captures_read(locations, subject) {
            None => "",
            Some(_) => locations
                .get(check_group_index(regex, idx)?)
                .map_or("", |(start, end)| &subject[start..end]),
        };
        builder.append_value(text);
    }
    Ok(Arc::new(builder.finish()))
}

fn check_group_index(regex: &Regex, idx: i32) -> Result<usize> {
    let group_count = regex.captures_len() - 1;
    usize::try_from(idx)
        .ok()
        .filter(|group| *group <= group_count)
        .ok_or_else(|| {
            exec_datafusion_err!(
                "[INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX] The value of parameter(s) \
                 `idx` in `{NAME}` is invalid: Expects group index between 0 and \
                 {group_count}, but got {idx}."
            )
        })
}

fn compile(pattern: &str) -> Result<Regex> {
    Regex::new(pattern).map_err(|_| {
        exec_datafusion_err!(
            "[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` in \
             `{NAME}` is invalid: '{pattern}'."
        )
    })
}
