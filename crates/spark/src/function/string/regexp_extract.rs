// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Int32Array, StringArray, StringBuilder};
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use regex::{CaptureLocations, Regex};

use crate::function::error_utils::invalid_arg_count_exec_err;

const NAME: &str = "regexp_extract";

/// Spark-compatible `regexp_extract(str, regexp[, idx])`.
/// <https://spark.apache.org/docs/latest/api/sql/index.html#regexp_extract>
///
/// Returns the text matched by group `idx` (default `1`, `0` is the whole match)
/// of the first match of `regexp` in `str`.
///
/// - NULL if any argument is NULL.
/// - An empty string if there is no match, or if the group did not take part in
///   the match.
/// - An error if `idx` is outside `0..=groupCount`
///   (`INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX`) or `regexp` is not a valid
///   pattern (`INVALID_PARAMETER_VALUE.PATTERN`).
///
/// Patterns use the Rust [`regex`] syntax, not `java.util.regex`. Java-only
/// features (lookaround, backreferences, possessive quantifiers) are rejected
/// as invalid patterns here, and shorthands such as `\d` match Unicode digits
/// where Java's match only ASCII.
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
        let num_rows = args.number_rows;
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let strings = arrays[0].as_string::<i32>();
        let patterns = arrays[1].as_string::<i32>();
        let default_idx;
        let indexes = match arrays.get(2) {
            Some(idx) => as_int32_array(idx)?,
            None => {
                default_idx = Int32Array::from(vec![1; num_rows]);
                &default_idx
            }
        };
        Ok(ColumnarValue::Array(extract(strings, patterns, indexes)?))
    }
}

fn extract(
    strings: &StringArray,
    patterns: &StringArray,
    indexes: &Int32Array,
) -> Result<ArrayRef> {
    let mut builder = StringBuilder::with_capacity(strings.len(), 0);
    // The pattern is almost always a literal, so keep only the last one compiled,
    // with a capture buffer reused across rows instead of allocated per match.
    let mut cached: Option<(&str, Regex, CaptureLocations)> = None;

    for row in 0..strings.len() {
        if strings.is_null(row) || patterns.is_null(row) || indexes.is_null(row) {
            builder.append_null();
            continue;
        }
        let pattern = patterns.value(row);
        let (_, regex, locations) = match &mut cached {
            Some(entry) if entry.0 == pattern => entry,
            _ => {
                let regex = compile(pattern)?;
                let locations = regex.capture_locations();
                cached.insert((pattern, regex, locations))
            }
        };
        let idx = indexes.value(row);
        let group_count = regex.captures_len() - 1;
        let group = usize::try_from(idx).ok().filter(|g| *g <= group_count);
        let Some(group) = group else {
            return Err(exec_datafusion_err!(
                "[INVALID_PARAMETER_VALUE.REGEX_GROUP_INDEX] The value of parameter(s) \
                 `idx` in `{NAME}` is invalid: Expects group index between 0 and \
                 {group_count}, but got {idx}."
            ));
        };
        let subject = strings.value(row);
        let text = regex
            .captures_read(locations, subject)
            .and_then(|_| locations.get(group))
            .map_or("", |(start, end)| &subject[start..end]);
        builder.append_value(text);
    }
    Ok(Arc::new(builder.finish()))
}

fn compile(pattern: &str) -> Result<Regex> {
    Regex::new(pattern).map_err(|_| {
        exec_datafusion_err!(
            "[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` in \
             `{NAME}` is invalid: '{pattern}'."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        strings: Vec<Option<&str>>,
        patterns: Vec<Option<&str>>,
        indexes: Vec<Option<i32>>,
    ) -> Result<Vec<Option<String>>> {
        let out = extract(
            &StringArray::from(strings),
            &StringArray::from(patterns),
            &Int32Array::from(indexes),
        )?;
        Ok(out
            .as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_string))
            .collect())
    }

    fn one(s: &str, p: &str, idx: i32) -> Result<String> {
        Ok(run(vec![Some(s)], vec![Some(p)], vec![Some(idx)])?
            .remove(0)
            .unwrap())
    }

    // Expected values come from Spark's regexp-functions.sql.out golden file.
    #[test]
    fn groups() -> Result<()> {
        assert_eq!(one("1a 2b 14m", r"\d+", 0)?, "1");
        assert_eq!(one("1a 2b 14m", r"(\d+)([a-z]+)", 0)?, "1a");
        assert_eq!(one("1a 2b 14m", r"(\d+)([a-z]+)", 1)?, "1");
        assert_eq!(one("1a 2b 14m", r"(\d+)([a-z]+)", 2)?, "a");
        Ok(())
    }

    #[test]
    fn unmatched_group_and_no_match_give_empty_string() -> Result<()> {
        assert_eq!(one("1a 2b 14m", r"(\d+)?", 1)?, "1");
        assert_eq!(one("a b m", r"(\d+)?", 1)?, "");
        assert_eq!(one("a b m", r"(\d+)?([a-z]+)", 1)?, "");
        assert_eq!(one("abc", r"(\d+)", 1)?, "");
        Ok(())
    }

    #[test]
    fn invalid_group_index_and_pattern() {
        for (p, idx) in [(r"\d+", 1), (r"\d+", 2), (r"\d+", -1), (r"(\d+)(a)", 3)] {
            let err = one("1a 2b 14m", p, idx).unwrap_err().to_string();
            assert!(err.contains("REGEX_GROUP_INDEX"), "{p} {idx}: {err}");
        }
        let err = one("x", "(unclosed", 0).unwrap_err().to_string();
        assert!(err.contains("INVALID_PARAMETER_VALUE.PATTERN"), "{err}");
    }

    #[test]
    fn nulls_and_per_row_patterns() -> Result<()> {
        let got = run(
            vec![Some("a1"), None, Some("b2"), Some("c3"), Some("d4")],
            vec![
                Some(r"(\d)"),
                Some(r"(\d)"),
                None,
                Some(r"([a-z])"),
                Some(r"(\d)"),
            ],
            vec![Some(1), Some(1), Some(1), Some(1), None],
        )?;
        let expected = [Some("1"), None, None, Some("c"), None];
        assert_eq!(got, expected.map(|s| s.map(str::to_string)));
        Ok(())
    }

    #[test]
    fn null_pattern_skips_validation() -> Result<()> {
        // Spark returns NULL before looking at an out-of-range idx on a NULL row.
        let got = run(vec![Some("a")], vec![None], vec![Some(99)])?;
        assert_eq!(got, vec![None]);
        Ok(())
    }
}
