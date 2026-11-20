// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use crate::function::nullability::{SparkNullability, spark_nullability};
use arrow::array::{Array, ArrayRef, BooleanArray, Scalar};
use arrow::compute::kernels::cast::cast;
use arrow::compute::kernels::comparison::contains as arrow_contains;
use arrow::datatypes::DataType;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;

/// Spark's `Contains` is a null-intolerant `StringPredicate`, a `BinaryExpression`.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

/// Spark-compatible `contains` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#contains>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns true if `right` is found inside `left`. DataFusion's own `contains` takes only \
        strings; this one also accepts binary. When both arguments are binary the match is on bytes. \
        When one argument is a string and the other is binary, both are compared as bytes, which is \
        the same as Spark casting the binary to a string. Integer, boolean and date arguments are \
        cast to a string first. Other types are rejected at planning, including floating-point, \
        decimal and timestamp arguments, which Spark casts to a string \
        (see https://github.com/apache/datafusion-spark/issues/347). The match is case-sensitive, and an empty `right` is found in every \
        value, including an empty `left`. Returns NULL if either input is NULL.",
    syntax_example = "contains(left, right)",
    sql_example = r#"```sql
> SELECT contains('Spark SQL', 'Spark');
true
> SELECT contains('Spark SQL', 'SPARK');
false
> SELECT contains(x'537061726b2053514c', x'537061726b');
true
```"#,
    argument(
        name = "left",
        description = "String or binary expression to search in. NULL returns NULL."
    ),
    argument(
        name = "right",
        description = "String or binary expression to search for. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkContains {
    signature: Signature,
}

impl Default for SparkContains {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkContains {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

/// How an argument type takes part in `contains`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Null,
    String,
    Binary,
}

/// Classifies an argument type, returning the type it is coerced to before
/// unification. Types Spark implicitly casts to a string and whose Arrow string
/// form matches Spark's become `Utf8`.
fn classify(name: &str, dt: &DataType) -> Result<(Kind, DataType)> {
    match dt {
        DataType::Null => Ok((Kind::Null, DataType::Null)),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            Ok((Kind::String, dt.clone()))
        }
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            Ok((Kind::Binary, dt.clone()))
        }
        DataType::FixedSizeBinary(_) => Ok((Kind::Binary, DataType::Binary)),
        DataType::Dictionary(_, value) => classify(name, value),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Boolean
        | DataType::Date32 => Ok((Kind::String, DataType::Utf8)),
        other => plan_err!(
            "Unsupported Data Type: Spark `{name}` function expects STRING or BINARY arguments, got {other}"
        ),
    }
}

/// The widest of two string (or two binary) types, so both sides of the Arrow
/// kernel have the same type.
fn unify(l: &DataType, r: &DataType, binary: bool) -> DataType {
    let (large, view, plain) = if binary {
        (
            DataType::LargeBinary,
            DataType::BinaryView,
            DataType::Binary,
        )
    } else {
        (DataType::LargeUtf8, DataType::Utf8View, DataType::Utf8)
    };
    let is = |t: &DataType| l == t || r == t;
    if is(&large) {
        large
    } else if is(&view) {
        view
    } else {
        plain
    }
}

impl ScalarUDFImpl for SparkContains {
    fn name(&self) -> &str {
        "contains"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [left, right] = take_function_args(self.name(), arg_types)?;
        let (l_kind, l_type) = classify(self.name(), left)?;
        let (r_kind, r_type) = classify(self.name(), right)?;

        // Spark matches bytes when both sides are binary. With a string and a
        // binary it casts the binary to a string, but a UTF8_BINARY string
        // compares as bytes too, so comparing both as binary gives the same
        // answer and does not fail on bytes that are not valid UTF-8.
        let target = if l_kind == Kind::Binary || r_kind == Kind::Binary {
            unify(&l_type, &r_type, true)
        } else {
            unify(&l_type, &r_type, false)
        };
        Ok(vec![target.clone(), target])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    spark_nullability!(NULLABILITY, return_type);

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [haystack, needle] = take_function_args(self.name(), &args.args)?;
        spark_contains(haystack, needle)
    }
}

/// Evaluates `contains`, keeping a scalar needle (or haystack) as an Arrow
/// `Scalar` so the kernel can reuse its searcher across rows.
fn spark_contains(
    haystack: &ColumnarValue,
    needle: &ColumnarValue,
) -> Result<ColumnarValue> {
    match (haystack, needle) {
        (ColumnarValue::Array(h), ColumnarValue::Array(n)) => {
            let n = align(n, h.data_type())?;
            Ok(ColumnarValue::Array(Arc::new(arrow_contains(h, &n)?)))
        }
        (ColumnarValue::Array(h), ColumnarValue::Scalar(n)) => {
            if n.is_null() {
                return Ok(null_array(h.len()));
            }
            let n = Scalar::new(scalar_to_array(n, h.data_type())?);
            Ok(ColumnarValue::Array(Arc::new(arrow_contains(h, &n)?)))
        }
        (ColumnarValue::Scalar(h), ColumnarValue::Array(n)) => {
            if h.is_null() {
                return Ok(null_array(n.len()));
            }
            let h = Scalar::new(scalar_to_array(h, n.data_type())?);
            Ok(ColumnarValue::Array(Arc::new(arrow_contains(&h, n)?)))
        }
        (ColumnarValue::Scalar(h), ColumnarValue::Scalar(n)) => {
            if h.is_null() || n.is_null() {
                return Ok(ColumnarValue::Scalar(ScalarValue::Boolean(None)));
            }
            let h = h.to_array()?;
            let n = scalar_to_array(n, h.data_type())?;
            let result = arrow_contains(&Scalar::new(h), &Scalar::new(n))?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        }
    }
}

fn null_array(len: usize) -> ColumnarValue {
    ColumnarValue::Array(Arc::new(BooleanArray::new_null(len)))
}

/// Materializes a scalar as a length-1 array of `target` type.
fn scalar_to_array(scalar: &ScalarValue, target: &DataType) -> Result<ArrayRef> {
    align(&scalar.to_array()?, target)
}

/// Casts `array` to `target` when the types differ. `coerce_types` gives both
/// sides the same type, so this only matters when the function is invoked
/// directly; it accepts string and binary types only.
fn align(array: &ArrayRef, target: &DataType) -> Result<ArrayRef> {
    if array.data_type() == target {
        return Ok(Arc::clone(array));
    }
    let is_supported = |dt: &DataType| {
        let dt = match dt {
            DataType::Dictionary(_, value) => value.as_ref(),
            dt => dt,
        };
        matches!(
            dt,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
        )
    };
    if !is_supported(array.data_type()) || !is_supported(target) {
        return exec_err!(
            "Unsupported Data Type: Spark `contains` function expects STRING or BINARY arguments, got {} and {}",
            target,
            array.data_type()
        );
    }
    Ok(cast(array, target)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, BinaryViewArray, DictionaryArray, LargeBinaryArray,
        LargeStringArray, StringArray, StringViewArray,
    };
    use arrow::datatypes::{Field, Int32Type};
    use datafusion_common::config::ConfigOptions;

    fn invoke(args: Vec<ColumnarValue>, number_rows: usize) -> Result<ColumnarValue> {
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(i, a)| Arc::new(Field::new(format!("a{i}"), a.data_type(), true)))
            .collect();
        SparkContains::new().invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new("r", DataType::Boolean, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn invoke_array(args: Vec<ColumnarValue>, number_rows: usize) -> BooleanArray {
        match invoke(args, number_rows).unwrap() {
            ColumnarValue::Array(a) => {
                a.as_any().downcast_ref::<BooleanArray>().unwrap().clone()
            }
            ColumnarValue::Scalar(s) => panic!("expected array, got {s:?}"),
        }
    }

    fn arr(a: impl Array + 'static) -> ColumnarValue {
        ColumnarValue::Array(Arc::new(a))
    }

    fn utf8(s: Option<&str>) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Utf8(s.map(str::to_string)))
    }

    #[test]
    fn test_contains_array_scalar() {
        let res = invoke_array(
            vec![
                arr(StringArray::from(vec![
                    Some("hello world"),
                    Some("foo bar"),
                    Some("testing"),
                    None,
                ])),
                utf8(Some("world")),
            ],
            4,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![Some(true), Some(false), Some(false), None])
        );
    }

    #[test]
    fn test_contains_array_array() {
        let res = invoke_array(
            vec![
                arr(StringArray::from(vec![
                    Some("hello world"),
                    Some("hello"),
                    None,
                    Some("abc"),
                    Some("abc"),
                ])),
                arr(StringArray::from(vec![
                    Some("world"),
                    Some(""),
                    Some("a"),
                    None,
                    Some("d"),
                ])),
            ],
            5,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![Some(true), Some(true), None, None, Some(false)])
        );
    }

    #[test]
    fn test_contains_scalar_scalar() {
        for (h, n, expected) in [
            (Some("hello world"), Some("world"), Some(true)),
            (Some("hello world"), Some("xyz"), Some(false)),
            (Some("Spark SQL"), Some("SPARK"), Some(false)),
            (Some(""), Some(""), Some(true)),
            (None, Some("a"), None),
            (Some("a"), None, None),
        ] {
            // number_rows > 1 must still produce a scalar result
            let res = invoke(vec![utf8(h), utf8(n)], 3).unwrap();
            match res {
                ColumnarValue::Scalar(s) => {
                    assert_eq!(s, ScalarValue::Boolean(expected), "{h:?} {n:?}")
                }
                ColumnarValue::Array(_) => panic!("expected scalar"),
            }
        }
    }

    #[test]
    fn test_contains_null_needle() {
        let res = invoke_array(
            vec![
                arr(StringArray::from(vec![
                    Some("hello world"),
                    Some("foo bar"),
                ])),
                utf8(None),
            ],
            2,
        );
        assert_eq!(res, BooleanArray::from(vec![None, None]));

        let res = invoke_array(
            vec![
                arr(StringArray::from(vec![
                    Some("hello world"),
                    Some("foo bar"),
                ])),
                ColumnarValue::Scalar(ScalarValue::Null),
            ],
            2,
        );
        assert_eq!(res, BooleanArray::from(vec![None, None]));
    }

    #[test]
    fn test_contains_empty_needle() {
        let res = invoke_array(
            vec![
                arr(StringArray::from(vec![Some("hello world"), Some("")])),
                utf8(Some("")),
            ],
            2,
        );
        assert_eq!(res, BooleanArray::from(vec![true, true]));
    }

    #[test]
    fn test_contains_scalar_array_null_haystack() {
        let res = invoke_array(
            vec![
                utf8(None),
                arr(StringArray::from(vec![
                    Some("hello world"),
                    Some("foo bar"),
                ])),
            ],
            2,
        );
        assert_eq!(res, BooleanArray::from(vec![None, None]));
    }

    #[test]
    fn test_contains_scalar_array() {
        let res = invoke_array(
            vec![
                utf8(Some("hello world")),
                arr(StringArray::from(vec![
                    Some("hello"),
                    Some("world"),
                    Some("foo"),
                    None,
                ])),
            ],
            4,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![Some(true), Some(true), Some(false), None])
        );
    }

    #[test]
    fn test_contains_scalar_large_utf8() {
        let res = invoke_array(
            vec![
                ColumnarValue::Scalar(ScalarValue::LargeUtf8(Some("abc".to_string()))),
                arr(LargeStringArray::from(vec![
                    Some("a"),
                    Some("bc"),
                    None,
                    Some(""),
                    Some("d"),
                ])),
            ],
            5,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                None,
                Some(true),
                Some(false)
            ])
        );
    }

    #[test]
    fn test_contains_scalar_utf8_view() {
        let res = invoke_array(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8View(Some("abc".to_string()))),
                arr(StringViewArray::from(vec![
                    Some("a"),
                    Some("bc"),
                    None,
                    Some(""),
                    Some("d"),
                ])),
            ],
            5,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                None,
                Some(true),
                Some(false)
            ])
        );
    }

    #[test]
    fn test_contains_scalar_dictionary() {
        let haystack = ScalarValue::Dictionary(
            Box::new(DataType::Int32),
            Box::new(ScalarValue::Utf8(Some("abc".to_string()))),
        );
        let needle = DictionaryArray::<Int32Type>::from_iter(vec![
            Some("a"),
            Some("bc"),
            None,
            Some(""),
            Some("d"),
        ]);
        let res = invoke_array(vec![ColumnarValue::Scalar(haystack), arr(needle)], 5);
        assert_eq!(
            res,
            BooleanArray::from(vec![
                Some(true),
                Some(true),
                None,
                Some(true),
                Some(false)
            ])
        );
    }

    #[test]
    fn test_contains_array_scalar_large_utf8_haystack() {
        // A Utf8 needle is aligned to a LargeUtf8 haystack.
        let res = invoke_array(
            vec![
                arr(LargeStringArray::from(vec![Some("abc"), Some("xyz")])),
                utf8(Some("bc")),
            ],
            2,
        );
        assert_eq!(res, BooleanArray::from(vec![true, false]));
    }

    #[test]
    fn test_contains_scalar_array_empty() {
        let res = invoke_array(
            vec![
                utf8(Some("hello world")),
                arr(StringArray::from(Vec::<Option<&str>>::new())),
            ],
            0,
        );
        assert_eq!(res.len(), 0);
    }

    #[test]
    fn test_contains_binary() {
        let haystack: Vec<Option<&[u8]>> =
            vec![Some(b"Spark SQL"), Some(b"\x00\xff\x01"), Some(b""), None];
        let needle: Vec<Option<&[u8]>> =
            vec![Some(b"Spark"), Some(b"\xff"), Some(b""), Some(b"a")];
        let expected = BooleanArray::from(vec![Some(true), Some(true), Some(true), None]);

        let res = invoke_array(
            vec![
                arr(BinaryArray::from(haystack.clone())),
                arr(BinaryArray::from(needle.clone())),
            ],
            4,
        );
        assert_eq!(res, expected);

        let res = invoke_array(
            vec![
                arr(LargeBinaryArray::from(haystack.clone())),
                arr(LargeBinaryArray::from(needle.clone())),
            ],
            4,
        );
        assert_eq!(res, expected);

        let res = invoke_array(
            vec![
                arr(BinaryViewArray::from(haystack.clone())),
                arr(BinaryViewArray::from(needle)),
            ],
            4,
        );
        assert_eq!(res, expected);

        let res = invoke_array(
            vec![
                arr(BinaryArray::from(haystack)),
                ColumnarValue::Scalar(ScalarValue::Binary(Some(b"\xff".to_vec()))),
            ],
            4,
        );
        assert_eq!(
            res,
            BooleanArray::from(vec![Some(false), Some(true), Some(false), None])
        );

        let res = invoke(
            vec![
                ColumnarValue::Scalar(ScalarValue::Binary(Some(b"Spark SQL".to_vec()))),
                ColumnarValue::Scalar(ScalarValue::Binary(Some(b"SQL".to_vec()))),
            ],
            2,
        )
        .unwrap();
        assert!(matches!(
            res,
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true)))
        ));
    }

    #[test]
    fn test_coerce_types() {
        let f = SparkContains::new();
        let c = |l: DataType, r: DataType| f.coerce_types(&[l, r]).unwrap();
        assert_eq!(
            c(DataType::Utf8, DataType::Utf8),
            vec![DataType::Utf8, DataType::Utf8]
        );
        assert_eq!(
            c(DataType::Utf8, DataType::LargeUtf8),
            vec![DataType::LargeUtf8, DataType::LargeUtf8]
        );
        assert_eq!(
            c(DataType::Utf8View, DataType::Null),
            vec![DataType::Utf8View, DataType::Utf8View]
        );
        assert_eq!(
            c(DataType::Binary, DataType::Utf8),
            vec![DataType::Binary, DataType::Binary]
        );
        assert_eq!(
            c(DataType::Utf8, DataType::BinaryView),
            vec![DataType::BinaryView, DataType::BinaryView]
        );
        assert_eq!(
            c(DataType::Int64, DataType::Int32),
            vec![DataType::Utf8, DataType::Utf8]
        );
        assert_eq!(
            c(DataType::Null, DataType::Null),
            vec![DataType::Utf8, DataType::Utf8]
        );
        assert!(
            f.coerce_types(&[DataType::Float64, DataType::Utf8])
                .is_err()
        );
        assert!(f.coerce_types(&[DataType::Utf8]).is_err());
    }

    #[test]
    fn test_contains_invalid_type_error() {
        let err = invoke(
            vec![
                ColumnarValue::Scalar(ScalarValue::Int32(Some(123))),
                arr(StringArray::from(vec![Some("1")])),
            ],
            1,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("expects STRING or BINARY"),
            "unexpected error: {err}"
        );
    }
}
