// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use arrow::array::{
    Array, ArrayRef, AsArray, Int32Array, StringArray, StringArrayType, StringBuilder,
    new_null_array,
};
use arrow::compute::{can_cast_types, cast};
use arrow::datatypes::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::utils::offset_span_len;
use datafusion_common::{
    DataFusionError, Result, ScalarValue, exec_datafusion_err, exec_err,
    internal_datafusion_err, internal_err, plan_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use regex::{CaptureLocations, Regex};
use regex_syntax::ast::{
    AssertionKind, Ast, ClassSetBinaryOp, ClassSetBinaryOpKind, ClassSetItem,
    ClassUnicode, ClassUnicodeKind, Flag, Flags, FlagsItemKind, GroupKind, Visitor,
};

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
        `regex` engine, not Java's `java.util.regex`. Look-around, backreferences, quoting with `\\Q...\\E`, \
        `\\Z`, `\\h`, atomic groups `(?>...)`, the `(?d)` flag, and patterns that exceed the Rust engine's \
        compiled-size limit raise an \"unsupported regex pattern\" error, where Spark evaluates them. Only unclosed \
        or unopened groups and classes, invalid class ranges and repetitions, and unknown flags are reported as \
        `INVALID_PARAMETER_VALUE.PATTERN`; any other pattern the Rust engine rejects is reported as unsupported, \
        including a few that Java rejects too, such as `\\y`. \
        Rust syntax that Java rejects is also `INVALID_PARAMETER_VALUE.PATTERN`: the `R` flag, `\\b{start}` and \
        the other `\\b{...}` word boundaries, `(?P<name>...)`, and group names that are not an ASCII letter \
        followed by ASCII letters and digits. Syntax that both engines accept but read differently is reported as \
        unsupported rather than evaluated to a different answer: the `U` flag (Java's UNICODE_CHARACTER_CLASS, \
        which swaps greediness in Rust), `[:alpha:]` and the other POSIX classes in brackets, the class operators \
        `--` and `~~`, `\\<` and `\\>`, and the names of Java's ASCII-only POSIX classes, such as `\\p{Alpha}` \
        and `\\p{Lower}`. Other Unicode property names that Java does not know, such as `\\p{Letter}` or \
        `\\p{Latin}`, are accepted, where Java rejects them. \
        Possessive quantifiers are read as nested greedy repetition (`a++` as `(a+)+`), so they can match \
        where Java's would not. \
        `\\d`, `\\w` and `\\s` match Unicode characters, where Java's match only ASCII. \
        `.` matches `\\r`, where Java's does not. `$` matches only at the very end of the input, where Java's also \
        matches before a final line terminator. `(?i)` folds case using Unicode rules (so `k` matches the Kelvin \
        sign), where Java folds only ASCII by default.",
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
#[derive(Debug)]
pub struct SparkRegexpExtract {
    signature: Signature,
    /// Literal patterns compiled by earlier batches. One instance serves every
    /// query in a session, so the cache is keyed by pattern.
    patterns: PatternCache,
}

// The cache does not affect the result, so two instances with the same
// signature are equal.
impl PartialEq for SparkRegexpExtract {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
    }
}

impl Eq for SparkRegexpExtract {}

impl Hash for SparkRegexpExtract {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
    }
}

/// The literal patterns kept compiled. When it is full it is cleared, which
/// bounds its size if many different literal patterns are used.
const PATTERN_CACHE_SIZE: usize = 64;

#[derive(Debug, Default)]
struct PatternCache {
    patterns: Mutex<HashMap<String, Regex>>,
    /// How many patterns were compiled, so a test can tell that a later batch
    /// read the cache rather than compiling and inserting the pattern again.
    #[cfg(test)]
    compiles: std::sync::atomic::AtomicUsize,
}

impl PatternCache {
    fn get_or_compile(&self, pattern: &str) -> Result<Regex> {
        let lock = || {
            self.patterns
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        };
        if let Some(regex) = lock().get(pattern) {
            return Ok(regex.clone());
        }
        #[cfg(test)]
        self.compiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let regex = compile(pattern)?;
        let mut patterns = lock();
        if patterns.len() >= PATTERN_CACHE_SIZE {
            patterns.clear();
        }
        patterns.insert(pattern.to_string(), regex.clone());
        Ok(regex)
    }
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
            patterns: PatternCache::default(),
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
        let pattern_column = match &args.args[1] {
            ColumnarValue::Scalar(_) => None,
            ColumnarValue::Array(array) => Some(to_utf8("regexp", array)?),
        };
        let pattern = match (&args.args[1], &pattern_column) {
            (_, Some(patterns)) => Pattern::Array(patterns.as_string::<i32>()),
            (ColumnarValue::Scalar(scalar), None) => match scalar_str(scalar)? {
                Some(pattern) => Pattern::Scalar(pattern),
                None => return null_result(len, all_scalar),
            },
            (ColumnarValue::Array(_), None) => unreachable!("converted above"),
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

        // A caller that builds the physical expression directly, such as Comet,
        // can pass any string type. Each result is a substring of its input, so
        // the bytes of the visible input values bound the output.
        let strings = args.args[0].to_array(len)?;
        let result = match strings.data_type() {
            DataType::Utf8 => {
                let strings = strings.as_string::<i32>();
                let capacity = offset_span_len(strings.offsets());
                extract(strings, capacity, pattern, index, &self.patterns)?
            }
            DataType::LargeUtf8 => {
                let strings = strings.as_string::<i64>();
                let capacity = offset_span_len(strings.offsets());
                extract(strings, capacity, pattern, index, &self.patterns)?
            }
            DataType::Utf8View => {
                let strings = strings.as_string_view();
                let capacity = strings.lengths().map(|len| len as usize).sum();
                extract(strings, capacity, pattern, index, &self.patterns)?
            }
            other => return exec_err!("{NAME} expects a string str, got {other}"),
        };
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

/// A pattern column as `Utf8`. Other string types are cast, and anything else
/// is an error.
fn to_utf8(name: &str, array: &ArrayRef) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Utf8 => Ok(Arc::clone(array)),
        DataType::LargeUtf8 | DataType::Utf8View => Ok(cast(array, &DataType::Utf8)?),
        other => exec_err!("{NAME} expects a string {name}, got {other}"),
    }
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

fn extract<'a, S: StringArrayType<'a> + Copy>(
    strings: S,
    capacity: usize,
    pattern: Pattern,
    index: Index,
    cache: &PatternCache,
) -> Result<ArrayRef> {
    let mut builder = StringBuilder::with_capacity(strings.len(), capacity);
    // A literal pattern comes from the cache on the first non-NULL row, so a NULL
    // row is not checked. A column pattern keeps only the last one compiled,
    // since it rarely changes per row. The capture buffer is reused across rows
    // instead of allocated per match.
    let mut cached: Option<(&str, Regex, CaptureLocations)> = None;

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
        let (false, Some(pattern_str), Some(idx)) =
            (strings.is_null(row), row_pattern, idx)
        else {
            builder.append_null();
            continue;
        };
        let (_, regex, locations) = match &mut cached {
            Some(entry) if entry.0 == pattern_str => entry,
            _ => {
                let regex = match pattern {
                    Pattern::Scalar(_) => cache.get_or_compile(pattern_str)?,
                    Pattern::Array(_) => compile(pattern_str)?,
                };
                let locations = regex.capture_locations();
                cached.insert((pattern_str, regex, locations))
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
                 {group_count}, but got {idx}. SQLSTATE: 22023"
            )
        })
}

fn compile(pattern: &str) -> Result<Regex> {
    let quoted = to_sql_value(pattern);
    let unsupported = |reason: String| {
        DataFusionError::NotImplemented(format!(
            "Unsupported regex pattern {quoted} in `{NAME}`: {reason}. \
             See https://github.com/apache/datafusion-spark/issues/255"
        ))
    };
    match java_difference(pattern) {
        None => {}
        Some(JavaDifference::Invalid) => {
            return Err(exec_datafusion_err!(
                "[INVALID_PARAMETER_VALUE.PATTERN] The value of parameter(s) `regexp` in \
                 `{NAME}` is invalid: {quoted}. SQLSTATE: 22023"
            ));
        }
        Some(JavaDifference::Unsupported(reason)) => return Err(unsupported(reason)),
    }
    // The pattern parsed, so a later stage of the Rust engine rejected it.
    Regex::new(pattern).map_err(|err| match err {
        regex::Error::CompiledTooBig(_) => unsupported(
            "the compiled pattern exceeds the Rust regex size limit".to_string(),
        ),
        err => unsupported(format!("the Rust regex engine rejects it ({err})")),
    })
}

/// How `java.util.regex`, which Spark uses, would treat a pattern differently
/// from the Rust regex engine.
#[derive(Debug)]
enum JavaDifference {
    /// Java rejects the pattern, so it is reported as an invalid pattern.
    Invalid,
    /// Java accepts the pattern but the Rust engine rejects it, or reads it
    /// differently, so it is reported as unsupported rather than evaluated to
    /// a different answer than Spark's.
    Unsupported(String),
}

/// Checks a pattern with `regex_syntax`, whose AST and error kinds are API,
/// rather than matching on the text of [`regex::Error`].
///
/// A pattern that does not parse is invalid only for the kinds of syntax error
/// that Java also raises. Anything else, such as `\Q...\E`, `\Z`, `\h`,
/// `(?>...)` or `(?d)`, is unsupported, which also covers a few patterns that
/// Java rejects, such as `\y`. A pattern that parses is walked for the
/// constructs that the two engines read differently.
fn java_difference(pattern: &str) -> Option<JavaDifference> {
    use regex_syntax::ast::ErrorKind;
    let ast_err = match regex_syntax::ast::parse::Parser::new().parse(pattern) {
        Ok(ast) => return regex_syntax::ast::visit(&ast, JavaDifferences).err(),
        Err(err) => err,
    };
    let unsupported = |reason: String| Some(JavaDifference::Unsupported(reason));
    match ast_err.kind() {
        ErrorKind::GroupUnclosed
        | ErrorKind::GroupUnopened
        | ErrorKind::ClassUnclosed
        | ErrorKind::ClassRangeInvalid
        | ErrorKind::RepetitionMissing
        | ErrorKind::RepetitionCountInvalid => Some(JavaDifference::Invalid),
        // Java also rejects an unknown flag, except `d` (UNIX_LINES) and `>`
        // (an atomic group).
        ErrorKind::FlagUnrecognized => {
            let span = ast_err.span();
            let flag = &pattern[span.start.offset..span.end.offset];
            if matches!(flag, "d" | ">") {
                unsupported(format!(
                    "the Rust regex engine does not support the flag '{flag}'"
                ))
            } else {
                Some(JavaDifference::Invalid)
            }
        }
        ErrorKind::UnsupportedLookAround => unsupported(
            "look-around is not supported by the Rust regex engine".to_string(),
        ),
        ErrorKind::UnsupportedBackreference => unsupported(
            "backreferences are not supported by the Rust regex engine".to_string(),
        ),
        kind => unsupported(format!(
            "the Rust regex engine does not support it ({kind})"
        )),
    }
}

/// Java's POSIX character classes, which match ASCII only (unless the `U` flag
/// is set, which is unsupported). The Rust engine reads the same names as
/// Unicode properties, so `\p{Alpha}` matches `é` there but not in Java.
const JAVA_POSIX_CLASSES: [&str; 13] = [
    "Lower", "Upper", "ASCII", "Alpha", "Digit", "Alnum", "Punct", "Graph", "Print",
    "Blank", "Cntrl", "XDigit", "Space",
];

/// Finds the constructs of a parsed pattern that Java rejects or reads
/// differently. Each is checked by its AST node, so an escaped or quoted
/// look-alike, such as `\[\[:alpha:\]\]`, is not reported.
struct JavaDifferences;

impl JavaDifferences {
    fn check_flags(flags: &Flags) -> Result<(), JavaDifference> {
        // Java has no R flag, even negated.
        if flags
            .items
            .iter()
            .any(|item| item.kind == FlagsItemKind::Flag(Flag::CRLF))
        {
            return Err(JavaDifference::Invalid);
        }
        // `(?-U)` is the default in both engines, so only setting it differs.
        if flags.flag_state(Flag::SwapGreed) == Some(true) {
            return Err(JavaDifference::Unsupported(
                "Java's flag 'U' is UNICODE_CHARACTER_CLASS, where the Rust regex \
                 engine's swaps greediness"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn check_unicode_class(class: &ClassUnicode) -> Result<(), JavaDifference> {
        match &class.kind {
            ClassUnicodeKind::Named(name)
                if JAVA_POSIX_CLASSES.contains(&name.as_str()) =>
            {
                let p = if class.negated { 'P' } else { 'p' };
                Err(JavaDifference::Unsupported(format!(
                    "Java's \\{p}{{{name}}} is an ASCII-only POSIX class, where the Rust \
                     regex engine's is a Unicode property"
                )))
            }
            _ => Ok(()),
        }
    }
}

impl Visitor for JavaDifferences {
    type Output = ();
    type Err = JavaDifference;

    fn finish(self) -> Result<(), JavaDifference> {
        Ok(())
    }

    fn visit_pre(&mut self, ast: &Ast) -> Result<(), JavaDifference> {
        match ast {
            Ast::Flags(set) => Self::check_flags(&set.flags),
            Ast::Group(group) => match &group.kind {
                GroupKind::NonCapturing(flags) => Self::check_flags(flags),
                // Java rejects `(?P<name>...)`, and a name that is not an ASCII
                // letter followed by ASCII letters and digits.
                GroupKind::CaptureName { starts_with_p, name } => {
                    let mut chars = name.name.chars();
                    let valid = !starts_with_p
                        && chars.next().is_some_and(|c| c.is_ascii_alphabetic())
                        && chars.all(|c| c.is_ascii_alphanumeric());
                    if valid {
                        Ok(())
                    } else {
                        Err(JavaDifference::Invalid)
                    }
                }
                GroupKind::CaptureIndex(_) => Ok(()),
            },
            Ast::Assertion(assertion) => match assertion.kind {
                // Java rejects `\b{start}`, `\b{end}`, `\b{start-half}` and
                // `\b{end-half}`.
                AssertionKind::WordBoundaryStart
                | AssertionKind::WordBoundaryEnd
                | AssertionKind::WordBoundaryStartHalf
                | AssertionKind::WordBoundaryEndHalf => Err(JavaDifference::Invalid),
                AssertionKind::WordBoundaryStartAngle
                | AssertionKind::WordBoundaryEndAngle => Err(JavaDifference::Unsupported(
                    "Java reads \\< and \\> as literal characters, where the Rust regex \
                     engine reads them as word boundaries"
                        .to_string(),
                )),
                AssertionKind::StartLine
                | AssertionKind::EndLine
                | AssertionKind::StartText
                | AssertionKind::EndText
                | AssertionKind::WordBoundary
                | AssertionKind::NotWordBoundary => Ok(()),
            },
            Ast::ClassUnicode(class) => Self::check_unicode_class(class),
            _ => Ok(()),
        }
    }

    fn visit_class_set_item_pre(
        &mut self,
        item: &ClassSetItem,
    ) -> Result<(), JavaDifference> {
        match item {
            ClassSetItem::Ascii(_) => Err(JavaDifference::Unsupported(
                "Java reads [:name:] inside a class as a nested class of its characters, \
                 where the Rust regex engine reads it as a POSIX class"
                    .to_string(),
            )),
            ClassSetItem::Unicode(class) => Self::check_unicode_class(class),
            _ => Ok(()),
        }
    }

    fn visit_class_set_binary_op_pre(
        &mut self,
        op: &ClassSetBinaryOp,
    ) -> Result<(), JavaDifference> {
        let operator = match op.kind {
            // `&&` is intersection in both engines.
            ClassSetBinaryOpKind::Intersection => return Ok(()),
            ClassSetBinaryOpKind::Difference => "--",
            ClassSetBinaryOpKind::SymmetricDifference => "~~",
        };
        Err(JavaDifference::Unsupported(format!(
            "Java has no class operator '{operator}', which it reads as literal characters"
        )))
    }
}

/// Quotes a string the way Spark's `toSQLValue` renders a string literal in an
/// error message: wrapped in single quotes, with `\` and `'` escaped.
fn to_sql_value(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{LargeStringArray, StringViewArray};
    use arrow::datatypes::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(
        udf: &SparkRegexpExtract,
        args: Vec<ColumnarValue>,
        number_rows: usize,
    ) -> Result<ColumnarValue> {
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("a", arg.data_type(), true)))
            .collect();
        udf.invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new(NAME, DataType::Utf8, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn pattern(pattern: &str) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(pattern.to_string())))
    }

    fn strings(result: ColumnarValue) -> Vec<Option<String>> {
        let array = result.into_array(1).unwrap();
        array
            .as_string::<i32>()
            .iter()
            .map(|s| s.map(str::to_string))
            .collect()
    }

    /// Ported from Comet's `large_utf8_subject_returns_utf8_array`: a caller
    /// that builds the physical expression directly can pass any string type,
    /// and the result is always `Utf8`.
    #[test]
    fn large_utf8_and_view_subjects_return_utf8() {
        let udf = SparkRegexpExtract::new();
        let subjects: [ArrayRef; 2] = [
            Arc::new(LargeStringArray::from(vec![Some("a1"), None, Some("b22")])),
            Arc::new(StringViewArray::from(vec![Some("a1"), None, Some("b22")])),
        ];
        for subject in subjects {
            let result = invoke(
                &udf,
                vec![ColumnarValue::Array(subject), pattern(r"(\d+)")],
                3,
            )
            .unwrap();
            assert_eq!(result.data_type(), DataType::Utf8);
            assert_eq!(
                strings(result),
                vec![Some("1".to_string()), None, Some("22".to_string())]
            );
        }
    }

    #[test]
    fn large_utf8_pattern_column() {
        let udf = SparkRegexpExtract::new();
        let subject: ArrayRef = Arc::new(StringArray::from(vec!["a1", "b2"]));
        let patterns: ArrayRef =
            Arc::new(LargeStringArray::from(vec![r"(\d)", "([a-z])"]));
        let result = invoke(
            &udf,
            vec![
                ColumnarValue::Array(subject),
                ColumnarValue::Array(patterns),
            ],
            2,
        )
        .unwrap();
        assert_eq!(
            strings(result),
            vec![Some("1".to_string()), Some("b".to_string())]
        );
    }

    #[test]
    fn non_string_subject_is_an_error() {
        let udf = SparkRegexpExtract::new();
        let subject: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let error = invoke(&udf, vec![ColumnarValue::Array(subject), pattern("(1)")], 1)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expects a string str, got Int32"),
            "{error}"
        );
    }

    /// The output is sized from the visible values, not the whole buffer that
    /// a slice shares with its parent.
    #[test]
    fn sliced_subject_allocates_for_the_slice() {
        let udf = SparkRegexpExtract::new();
        let padding = "x".repeat(65536);
        let parent = StringArray::from(vec![padding.as_str(), "a1", padding.as_str()]);
        for (offset, len, expected) in
            [(1, 0, vec![]), (1, 1, vec![Some("1".to_string())])]
        {
            let subject: ArrayRef = Arc::new(parent.slice(offset, len));
            let result = invoke(
                &udf,
                vec![ColumnarValue::Array(subject), pattern(r"(\d)")],
                len,
            )
            .unwrap()
            .into_array(len)
            .unwrap();
            let result = result.as_string::<i32>();
            assert_eq!(
                result
                    .iter()
                    .map(|s| s.map(str::to_string))
                    .collect::<Vec<_>>(),
                expected
            );
            assert!(
                result.values().capacity() < 1024,
                "allocated {} bytes",
                result.values().capacity()
            );
        }
    }

    #[test]
    fn literal_patterns_are_cached_across_batches() {
        let udf = SparkRegexpExtract::new();
        for subject in ["a1", "b2"] {
            let subject =
                ColumnarValue::Array(Arc::new(StringArray::from(vec![subject])));
            invoke(&udf, vec![subject, pattern(r"(\d)")], 1).unwrap();
        }
        // Counting compiles, not cache entries, since a cache that is never
        // read would also end up with one entry for the pattern.
        assert_eq!(
            udf.patterns
                .compiles
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert!(udf.patterns.patterns.lock().unwrap().contains_key(r"(\d)"));
    }
}
