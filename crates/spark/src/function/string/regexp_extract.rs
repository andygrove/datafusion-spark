// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Int32Array, StringArray, StringBuilder, new_null_array,
};
use arrow::compute::can_cast_types;
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::{
    DataFusionError, Result, ScalarValue, exec_datafusion_err, internal_datafusion_err,
    internal_err, plan_err,
};
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
        (`INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX`, as in Spark 3.5 and later; Spark 3.4 raised \
        `_LEGACY_ERROR_TEMP_2006`/`2007`) on a row that matches, and an empty string on a row that does not. \
        An invalid pattern is an error (`INVALID_PARAMETER_VALUE.PATTERN`). Returns NULL if any argument is NULL, \
        and a NULL row is not checked, so an invalid pattern is only reported on a row it is applied to. \
        As in Spark, `str` and `regexp` are cast to strings and `idx` to INT. \
        Known differences from Spark (https://github.com/apache/datafusion-spark/issues/255): patterns use the Rust \
        `regex` engine, not Java's `java.util.regex`. Look-around, backreferences, and patterns that exceed the Rust \
        engine's compiled-size limit raise an \"unsupported regex pattern\" error, where Spark evaluates them. \
        Possessive quantifiers are read as nested greedy repetition (`a++` as `(a+)+`), so they can match \
        where Java's would not. \
        `\\d`, `\\w` and `\\s` match Unicode characters, where Java's match only ASCII. \
        `.` matches `\\r`, where Java's does not. `$` matches only at the very end of the input, where Java's also \
        matches before a final line terminator. `(?i)` folds case using Unicode rules (so `k` matches the Kelvin \
        sign), where Java folds only ASCII by default. Rust's `(?P<name>...)` named groups are accepted, where Java \
        rejects them as an invalid pattern.",
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
        description = "String expression to search. Values of other types, such as numbers and binary, are cast to a string. NULL returns NULL."
    ),
    argument(
        name = "regexp",
        description = "String regular expression in Rust `regex` syntax. Values of other types are cast to a string. NULL returns NULL."
    ),
    argument(
        name = "idx",
        description = "Optional capture group index, cast to INT (numeric and string values are accepted). Defaults to 1; 0 is the whole match. NULL returns NULL."
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
        // Like Spark's implicit casts: str and regexp are cast to strings, idx to INT.
        let castable_to_string = |t: &DataType| {
            *t == DataType::Null || (!t.is_nested() && can_cast_types(t, &DataType::Utf8))
        };
        for (name, arg_type) in ["str", "regexp"].iter().zip(arg_types) {
            if !castable_to_string(arg_type) {
                return plan_err!("{NAME} expects a string {name}, got {arg_type}");
            }
        }
        let mut coerced = vec![DataType::Utf8, DataType::Utf8];
        if let Some(idx) = arg_types.get(2) {
            let castable_to_int = idx.is_numeric()
                || matches!(
                    idx,
                    DataType::Null
                        | DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                );
            if !castable_to_int {
                return plan_err!("{NAME} expects an INT idx, got {idx}");
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
                return internal_err!(
                    "{NAME} expects an INT idx, got {}",
                    other.data_type()
                );
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
            internal_datafusion_err!(
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
    Regex::new(pattern).map_err(|err| {
        let pattern = to_sql_value(pattern);
        match unsupported_reason(&err) {
            Some(reason) => DataFusionError::NotImplemented(format!(
                "Unsupported regex pattern {pattern} in `{NAME}`: {reason}. \
                 See https://github.com/apache/datafusion-spark/issues/255"
            )),
            None => exec_datafusion_err!(
                "[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` in \
                 `{NAME}` is invalid: {pattern}."
            ),
        }
    })
}

/// Returns why a pattern the Rust engine rejects may still be valid in Java,
/// for the errors the `regex` crate reports for features it does not support.
fn unsupported_reason(err: &regex::Error) -> Option<&'static str> {
    match err {
        regex::Error::CompiledTooBig(_) => {
            Some("the compiled pattern exceeds the Rust regex size limit")
        }
        regex::Error::Syntax(msg)
            if msg.contains(
                "look-around, including look-ahead and look-behind, is not supported",
            ) =>
        {
            Some("look-around is not supported by the Rust regex engine")
        }
        regex::Error::Syntax(msg) if msg.contains("backreferences are not supported") => {
            Some("backreferences are not supported by the Rust regex engine")
        }
        _ => None,
    }
}

/// Quotes a string the way Spark's `toSQLValue` renders a string literal in an
/// error message: wrapped in single quotes, with `\` and `'` escaped.
fn to_sql_value(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}
