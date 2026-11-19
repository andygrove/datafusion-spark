// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible `array_max` and `array_min`.

use crate::function::float_normalize::normalize_array;
use crate::function::type_coercion::{is_orderable, spark_sql_type};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, GenericListArray, OffsetSizeTrait,
    PrimitiveArray, UInt64Array, make_comparator, new_null_array,
};
use arrow::compute::{SortOptions, take};
use arrow::datatypes::{
    DataType, Date32Type, Decimal128Type, Field, FieldRef, Float32Type, Float64Type,
    Int8Type, Int16Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use std::cmp::Ordering;
use std::sync::Arc;

/// Spark-compatible `array_max` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_max>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the maximum element of the array, using Spark's ordering. NULL \
        elements are skipped, and an empty array or an array of only NULL elements returns \
        NULL. For floating point values, also inside nested arrays and structs, NaN is \
        greater than every other value, including infinity, and -0.0 is equal to 0.0. On a \
        tie the first of the tied elements is returned as it is, so the sign of a zero and the \
        bits of a NaN are kept. Strings and binary values compare bytewise. Arrays compare \
        element by element and then by length, and structs compare field by field; NULLs \
        nested inside them sort before other values. Map and INTERVAL elements, at any \
        depth, are an error, as in Spark. DataFusion's built-in of the same name differs: it \
        orders -0.0 before 0.0 rather than returning the first of them, sorts NULLs nested \
        inside arrays and structs after other values, rejects a NULL argument at planning \
        instead of returning NULL, and fails at execution on map elements.",
    syntax_example = "array_max(array)",
    sql_example = r#"```sql
> SELECT array_max(array(1, 20, NULL, 3));
20
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayMax {
    signature: Signature,
}

impl Default for SparkArrayMax {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayMax {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayMax {
    fn name(&self) -> &str {
        "array_max"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        element_type(self.name(), arg_types)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_array(self.name(), arg_types)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke(self.name(), args, Ordering::Greater)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Spark-compatible `array_min` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_min>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the minimum element of the array, using Spark's ordering. NULL \
        elements are skipped, and an empty array or an array of only NULL elements returns \
        NULL. For floating point values, also inside nested arrays and structs, NaN is \
        greater than every other value, including infinity, and -0.0 is equal to 0.0. On a \
        tie the first of the tied elements is returned as it is, so the sign of a zero and the \
        bits of a NaN are kept. Strings and binary values compare bytewise. Arrays compare \
        element by element and then by length, and structs compare field by field; NULLs \
        nested inside them sort before other values. Map and INTERVAL elements, at any \
        depth, are an error, as in Spark. DataFusion's built-in of the same name differs: it \
        orders -0.0 before 0.0 rather than returning the first of them, sorts NULLs nested \
        inside arrays and structs after other values, rejects a NULL argument at planning \
        instead of returning NULL, and fails at execution on map elements.",
    syntax_example = "array_min(array)",
    sql_example = r#"```sql
> SELECT array_min(array(1, 20, NULL, 3));
1
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayMin {
    signature: Signature,
}

impl Default for SparkArrayMin {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayMin {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayMin {
    fn name(&self) -> &str {
        "array_min"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        element_type(self.name(), arg_types)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_array(self.name(), arg_types)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke(self.name(), args, Ordering::Less)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// The element type of the array argument. A NULL-typed argument, which Spark accepts,
/// returns the NULL type.
fn element_type(name: &str, arg_types: &[DataType]) -> Result<DataType> {
    let [array] = take_function_args(name, arg_types)?;
    match array {
        DataType::Null => Ok(DataType::Null),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => Ok(field.data_type().clone()),
        other => unexpected_input_type(name, other),
    }
}

/// Accepts the argument types that Spark's `ArrayMax` and `ArrayMin` accept: an array of an
/// orderable type, or NULL. Fixed-size lists are cast to lists, and dictionary-encoded
/// elements to their value type.
fn coerce_array(name: &str, arg_types: &[DataType]) -> Result<Vec<DataType>> {
    let [array] = take_function_args(name, arg_types)?;
    let coerced = match array {
        DataType::Null => DataType::Null,
        DataType::List(field) => DataType::List(decode_field(field)),
        DataType::LargeList(field) => DataType::LargeList(decode_field(field)),
        DataType::FixedSizeList(field, _) => DataType::List(Arc::new(Field::new(
            field.name(),
            decode_dictionaries(field.data_type()),
            true,
        ))),
        other => return unexpected_input_type(name, other),
    };
    let element = element_type(name, std::slice::from_ref(&coerced))?;
    if !is_orderable(&element) {
        return plan_err!(
            "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{name}` does not support \
             ordering on type \"{}\". SQLSTATE: 42K09",
            spark_sql_type(&element)
        );
    }
    Ok(vec![coerced])
}

/// Replaces every dictionary type, at any depth of list and struct, with its value type.
/// Spark has no dictionary types, but engines built on these functions pass
/// dictionary-encoded columns. The typed path and the float normalization of the generic
/// path do not look inside dictionaries, so `-0.0` would otherwise order below `0.0`.
fn decode_dictionaries(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value) => decode_dictionaries(value),
        DataType::List(field) => DataType::List(decode_field(field)),
        DataType::LargeList(field) => DataType::LargeList(decode_field(field)),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(decode_field(field), *size)
        }
        DataType::Struct(fields) => {
            DataType::Struct(fields.iter().map(decode_field).collect())
        }
        other => other.clone(),
    }
}

fn decode_field(field: &FieldRef) -> FieldRef {
    Arc::new(
        field
            .as_ref()
            .clone()
            .with_data_type(decode_dictionaries(field.data_type())),
    )
}

fn unexpected_input_type<T>(name: &str, data_type: &DataType) -> Result<T> {
    plan_err!(
        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `{name}` \
         requires the \"ARRAY\" type, however it has the type \"{}\". SQLSTATE: 42K09",
        spark_sql_type(data_type)
    )
}

fn invoke(
    name: &str,
    args: ScalarFunctionArgs,
    target: Ordering,
) -> Result<ColumnarValue> {
    let [arg] = take_function_args(name, &args.args)?;
    let element_type = args.return_field.data_type();
    let array = match arg {
        ColumnarValue::Array(array) => Arc::clone(array),
        ColumnarValue::Scalar(scalar) => scalar.to_array()?,
    };

    let result = match array.data_type() {
        DataType::Null => new_null_array(element_type, array.len()),
        // All the elements are NULL, so every result is NULL.
        _ if element_type.is_null() => new_null_array(element_type, array.len()),
        DataType::List(_) => extrema(array.as_list::<i32>(), target)?,
        DataType::LargeList(_) => extrema(array.as_list::<i64>(), target)?,
        other => return exec_err!("{name} does not support type {other}"),
    };

    if matches!(arg, ColumnarValue::Scalar(_)) {
        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
            &result, 0,
        )?))
    } else {
        Ok(ColumnarValue::Array(result))
    }
}

/// Returns, for each list, its first non-NULL element that no other element orders
/// `target` of, or NULL when the list is NULL, empty, or holds only NULL elements.
fn extrema<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    target: Ordering,
) -> Result<ArrayRef> {
    let offsets = list.offsets();
    // Only the values that this (possibly sliced) list refers to are compared.
    let base = offsets[0].as_usize();
    let len = offsets[offsets.len() - 1].as_usize() - base;
    let values = list.values().slice(base, len);

    // Primitive elements take a typed path: no copy of the values, no type-erased comparator
    // and no `take`.
    macro_rules! primitive {
        ($t:ty, $cmp:expr) => {
            return Ok(Arc::new(primitive_extrema::<$t, O>(
                list,
                values.as_primitive::<$t>(),
                base,
                target,
                $cmp,
            )))
        };
    }
    macro_rules! ordered {
        ($t:ty) => {
            primitive!($t, |a, b| a.cmp(&b))
        };
    }
    match values.data_type() {
        DataType::Int8 => ordered!(Int8Type),
        DataType::Int16 => ordered!(Int16Type),
        DataType::Int32 => ordered!(Int32Type),
        DataType::Int64 => ordered!(Int64Type),
        DataType::Decimal128(_, _) => ordered!(Decimal128Type),
        DataType::Date32 => ordered!(Date32Type),
        DataType::Timestamp(TimeUnit::Second, _) => ordered!(TimestampSecondType),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            ordered!(TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            ordered!(TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            ordered!(TimestampNanosecondType)
        }
        DataType::Float32 => primitive!(Float32Type, spark_float_cmp),
        DataType::Float64 => primitive!(Float64Type, spark_float_cmp),
        _ => {}
    }

    generic_extrema(list, &values, base, target)
}

/// [`extrema`] for any orderable element type. `values` is the list's values sliced to
/// start at offset `base`.
fn generic_extrema<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    values: &ArrayRef,
    base: usize,
    target: Ordering,
) -> Result<ArrayRef> {
    let offsets = list.offsets();
    // Compare canonicalized floats, so that NaN is the greatest value and -0.0 equals 0.0 at
    // any depth, but return the original element.
    let normalized = normalize_array(values)?;
    // Spark orders NULLs nested inside arrays and structs before other values.
    let cmp = make_comparator(
        normalized.as_ref(),
        normalized.as_ref(),
        SortOptions::default(),
    )?;

    let indices: UInt64Array = offsets
        .windows(2)
        .enumerate()
        .map(|(row, w)| {
            if list.is_null(row) {
                return None;
            }
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            (start..end)
                .filter(|&i| values.is_valid(i))
                .reduce(|best, i| if cmp(i, best) == target { i } else { best })
                .map(|i| i as u64)
        })
        .collect();

    Ok(take(values.as_ref(), &indices, None)?)
}

/// Spark's float ordering: NaN is equal to NaN and greater than every other value, and
/// -0.0 is equal to 0.0 (as under `partial_cmp`).
fn spark_float_cmp<F: PartialOrd>(a: F, b: F) -> Ordering {
    // `partial_cmp` fails only when a NaN is involved; `x != x` holds only for NaN.
    #[expect(clippy::eq_op)]
    a.partial_cmp(&b).unwrap_or_else(|| (a != a).cmp(&(b != b)))
}

/// [`extrema`] for primitive elements. `values` is the list's values sliced to start at
/// offset `base`. The winning value is copied unchanged, so the sign of a zero and the bits
/// of a NaN are kept, and the result keeps the element type's precision, scale or time zone.
fn primitive_extrema<T, O>(
    list: &GenericListArray<O>,
    values: &PrimitiveArray<T>,
    base: usize,
    target: Ordering,
    cmp: impl Fn(T::Native, T::Native) -> Ordering,
) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
    O: OffsetSizeTrait,
{
    let raw = values.values();
    let result: PrimitiveArray<T> = list
        .offsets()
        .windows(2)
        .enumerate()
        .map(|(row, w)| {
            if list.is_null(row) {
                return None;
            }
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            let mut best: Option<T::Native> = None;
            for i in start..end {
                if values.is_null(i) {
                    continue;
                }
                let v = raw[i];
                best = match best {
                    Some(b) if cmp(v, b) != target => Some(b),
                    _ => Some(v),
                };
            }
            best
        })
        .collect();
    result.with_data_type(values.data_type().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        BinaryArray, Float32Array, Float64Array, Int32Array, LargeListArray, ListArray,
        StringArray, StructArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Float32Type, Float64Type, Int32Type};
    use datafusion_common::config::ConfigOptions;

    fn run(udf: &dyn ScalarUDFImpl, input: ColumnarValue) -> ColumnarValue {
        let input_type = input.data_type();
        let return_type = udf.return_type(std::slice::from_ref(&input_type)).unwrap();
        let number_rows = match &input {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        udf.invoke_with_args(ScalarFunctionArgs {
            args: vec![input],
            arg_fields: vec![Arc::new(Field::new("input", input_type, true))],
            number_rows,
            return_field: Arc::new(Field::new("result", return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
        .unwrap()
    }

    fn max(input: &dyn Array) -> ArrayRef {
        run_array(&SparkArrayMax::new(), input)
    }

    fn min(input: &dyn Array) -> ArrayRef {
        run_array(&SparkArrayMin::new(), input)
    }

    fn run_array(udf: &dyn ScalarUDFImpl, input: &dyn Array) -> ArrayRef {
        let input = input.slice(0, input.len());
        match run(udf, ColumnarValue::Array(input)) {
            ColumnarValue::Array(result) => result,
            ColumnarValue::Scalar(_) => panic!("array input must produce array output"),
        }
    }

    fn list(values: ArrayRef, offsets: &[i32]) -> ListArray {
        ListArray::new(
            Arc::new(Field::new_list_field(values.data_type().clone(), true)),
            OffsetBuffer::new(offsets.to_vec().into()),
            values,
            None,
        )
    }

    fn f64_bits(array: &ArrayRef) -> Vec<Option<u64>> {
        array
            .as_primitive::<Float64Type>()
            .iter()
            .map(|v| v.map(f64::to_bits))
            .collect()
    }

    #[test]
    fn integers_skip_nulls() {
        let input = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(20), None, Some(3)]),
            Some(vec![]),
            Some(vec![None, None]),
            None,
            Some(vec![Some(-5)]),
        ]);
        assert_eq!(
            max(&input).as_ref(),
            &Int32Array::from(vec![Some(20), None, None, None, Some(-5)])
        );
        assert_eq!(
            min(&input).as_ref(),
            &Int32Array::from(vec![Some(1), None, None, None, Some(-5)])
        );
    }

    macro_rules! float_tests {
        ($name:ident, $arrow_type:ty, $array:ty, $native:ident, $positive:expr, $negative:expr) => {
            #[test]
            fn $name() {
                let positive = $native::from_bits($positive);
                let negative = $native::from_bits($negative);
                let mut long = vec![Some(1.0); 67];
                long[7] = Some(negative);
                long[8] = Some(positive);
                // (input, min, max)
                let cases = [
                    (vec![Some(0.0), Some(-0.0)], Some(0.0), Some(0.0)),
                    (vec![None, Some(-0.0), Some(0.0)], Some(-0.0), Some(-0.0)),
                    (
                        vec![Some(positive), Some(negative)],
                        Some(positive),
                        Some(positive),
                    ),
                    (
                        vec![Some(negative), Some(positive)],
                        Some(negative),
                        Some(negative),
                    ),
                    (
                        vec![
                            Some(negative),
                            Some($native::INFINITY),
                            Some($native::NEG_INFINITY),
                        ],
                        Some($native::NEG_INFINITY),
                        Some(negative),
                    ),
                    (vec![Some(3.0), None, Some(-2.0)], Some(-2.0), Some(3.0)),
                    (long, Some(1.0), Some(negative)),
                ];
                for (row, expected_min, expected_max) in cases {
                    let input =
                        ListArray::from_iter_primitive::<$arrow_type, _, _>([Some(row)]);
                    for (result, expected) in
                        [(min(&input), expected_min), (max(&input), expected_max)]
                    {
                        let result = result.as_any().downcast_ref::<$array>().unwrap();
                        assert_eq!(
                            result.iter().next().unwrap().map($native::to_bits),
                            expected.map($native::to_bits),
                        );
                    }
                }
            }
        };
    }

    float_tests!(
        float32_nan_and_signed_zero,
        Float32Type,
        Float32Array,
        f32,
        0x7fc0_0001,
        0xffc0_0002
    );
    float_tests!(
        float64_nan_and_signed_zero,
        Float64Type,
        Float64Array,
        f64,
        0x7ff8_0000_0000_0001,
        0xfff8_0000_0000_0002
    );

    #[test]
    fn scalar_and_sliced_input() {
        let input = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(99.0)]),
            Some(vec![Some(-0.0), None, Some(0.0)]),
            None,
        ])
        .slice(1, 2);
        for udf in [
            &SparkArrayMax::new() as &dyn ScalarUDFImpl,
            &SparkArrayMin::new(),
        ] {
            assert_eq!(
                f64_bits(&run_array(udf, &input)),
                vec![Some((-0.0f64).to_bits()), None]
            );
            let scalar = ScalarValue::try_from_array(&input, 0).unwrap();
            let ColumnarValue::Scalar(result) = run(udf, ColumnarValue::Scalar(scalar))
            else {
                panic!("scalar input must produce scalar output")
            };
            assert_eq!(
                f64_bits(&result.to_array().unwrap()),
                vec![Some((-0.0f64).to_bits())]
            );
        }
    }

    #[test]
    fn strings_and_binary_compare_bytewise() {
        let values: ArrayRef =
            Arc::new(StringArray::from(vec!["a", "b", "B", "", "é", "z"]));
        let input = list(values, &[0, 4, 6]);
        assert_eq!(max(&input).as_ref(), &StringArray::from(vec!["b", "é"]));
        assert_eq!(min(&input).as_ref(), &StringArray::from(vec!["", "z"]));

        let values: ArrayRef = Arc::new(BinaryArray::from(vec![
            &[0x01][..],
            &[0xff][..],
            &[0x01, 0x00][..],
        ]));
        let input = list(values, &[0, 3]);
        assert_eq!(max(&input).as_ref(), &BinaryArray::from(vec![&[0xff][..]]));
        assert_eq!(min(&input).as_ref(), &BinaryArray::from(vec![&[0x01][..]]));
    }

    #[test]
    fn large_list() {
        let input = LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(3), None, Some(7)]),
            None,
        ]);
        assert_eq!(max(&input).as_ref(), &Int32Array::from(vec![Some(7), None]));
        assert_eq!(min(&input).as_ref(), &Int32Array::from(vec![Some(3), None]));
    }

    #[test]
    fn nested_lists_order_lexicographically_with_nulls_first() {
        let children = ListArray::from_iter_primitive::<Float64Type, _, _>([
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(1.0), Some(2.0), Some(3.0)]),
            Some(vec![Some(2.0)]),
            Some(vec![]),
            Some(vec![Some(1.0), None]),
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(f64::NAN)]),
            Some(vec![Some(f64::INFINITY)]),
            None,
            Some(vec![Some(-0.0)]),
            Some(vec![Some(0.0)]),
        ]);
        let input = list(Arc::new(children.clone()), &[0, 4, 6, 9, 11]);
        let expected = |winners: [u64; 4]| -> ArrayRef {
            take(&children, &UInt64Array::from(winners.to_vec()), None).unwrap()
        };
        let result = max(&input);
        assert_eq!(&result, &expected([2, 5, 6, 9]));
        let result = min(&input);
        assert_eq!(&result, &expected([3, 4, 7, 9]));
        // The first tied winner keeps the sign of its zero.
        let last = result.as_list::<i32>().value(3);
        assert!(
            last.as_primitive::<Float64Type>()
                .value(0)
                .is_sign_negative()
        );
    }

    #[test]
    fn structs_compare_field_by_field() {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Float64Array::from(vec![
                Some(-0.0),
                Some(0.0),
                Some(1.0),
                None,
                Some(5.0),
                Some(5.0),
            ])),
            Arc::new(Int32Array::from(vec![2, 1, 1, 100, 1, 2])),
        ];
        let fields = vec![
            Arc::new(Field::new("a", DataType::Float64, true)),
            Arc::new(Field::new("b", DataType::Int32, true)),
        ];
        let children = StructArray::new(
            fields.into(),
            columns,
            Some(NullBuffer::from(vec![true, true, true, true, false, true])),
        );
        let input = list(Arc::new(children.clone()), &[0, 2, 4, 6]);
        let expected = |winners: [u64; 3]| {
            take(&children, &UInt64Array::from(winners.to_vec()), None).unwrap()
        };
        // -0.0 equals 0.0, so the second field decides. A NULL field sorts first, and a NULL
        // struct is skipped.
        assert_eq!(&max(&input), &expected([0, 2, 5]));
        assert_eq!(&min(&input), &expected([1, 3, 5]));
    }

    #[test]
    fn null_input_and_null_elements() {
        let input = new_null_array(&DataType::Null, 2);
        assert_eq!(max(&input).as_ref(), &new_null_array(&DataType::Null, 2));
        let input = list(new_null_array(&DataType::Null, 3), &[0, 0, 3]);
        assert_eq!(min(&input).as_ref(), &new_null_array(&DataType::Null, 2));
    }

    #[test]
    fn rejects_unsupported_types() {
        let udf = SparkArrayMax::new();
        let err = udf.coerce_types(&[DataType::Int32]).unwrap_err();
        assert!(err.to_string().contains("UNEXPECTED_INPUT_TYPE"), "{err}");
        let map = DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("key", DataType::Int32, false),
                        Field::new("value", DataType::Int32, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        );
        let array = DataType::new_list(DataType::new_list(map, true), true);
        let err = udf.coerce_types(&[array]).unwrap_err();
        assert!(err.to_string().contains("INVALID_ORDERING_TYPE"), "{err}");
        let fixed = DataType::new_fixed_size_list(DataType::Int32, 2, true);
        assert_eq!(
            udf.coerce_types(&[fixed]).unwrap(),
            vec![DataType::new_list(DataType::Int32, true)]
        );
    }

    /// Runs the typed path (through the UDF) and the generic path on the same input and
    /// checks that they pick bit-identical values.
    fn assert_paths_agree(input: &ListArray) {
        let values = {
            let offsets = input.offsets();
            let base = offsets[0] as usize;
            let len = offsets[offsets.len() - 1] as usize - base;
            (base, input.values().slice(base, len))
        };
        for (udf, target) in [
            (
                &SparkArrayMax::new() as &dyn ScalarUDFImpl,
                Ordering::Greater,
            ),
            (&SparkArrayMin::new(), Ordering::Less),
        ] {
            let fast = run_array(udf, input);
            let generic = generic_extrema(input, &values.1, values.0, target).unwrap();
            assert_eq!(f64_bits(&fast), f64_bits(&generic), "{}", udf.name());
        }
    }

    #[test]
    fn typed_float_path_matches_generic_path() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let specials = [
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0001),
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            1.0,
            -1.0,
        ];
        let rng = &mut StdRng::seed_from_u64(7);
        let rows: Vec<Option<Vec<Option<f64>>>> = (0..2000)
            .map(|_| {
                (rng.random_range(0..10) != 0).then(|| {
                    (0..rng.random_range(0..8))
                        .map(|_| {
                            (rng.random_range(0..5) != 0)
                                .then(|| specials[rng.random_range(0..specials.len())])
                        })
                        .collect()
                })
            })
            .collect();
        let input = ListArray::from_iter_primitive::<Float64Type, _, _>(rows);
        assert_paths_agree(&input);
        assert_paths_agree(&input.slice(13, 1500));
    }

    #[test]
    fn typed_path_keeps_element_type() {
        use arrow::array::{Decimal128Array, TimestampMicrosecondArray};
        let decimal_type = DataType::Decimal128(20, 3);
        let decimals: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(1_500), None, Some(-2_000), Some(1_499)])
                .with_data_type(decimal_type.clone()),
        );
        let timestamp_type =
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let timestamps: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(5), Some(9), None, Some(1)])
                .with_data_type(timestamp_type.clone()),
        );
        for values in [decimals, timestamps] {
            let data_type = values.data_type().clone();
            let input = ListArray::new(
                Arc::new(Field::new_list_field(data_type.clone(), true)),
                OffsetBuffer::from_lengths([4]),
                values,
                None,
            );
            let (max, min) = (max(&input), min(&input));
            assert_eq!(max.data_type(), &data_type);
            assert_eq!(min.data_type(), &data_type);
            if data_type == decimal_type {
                assert_eq!(max.as_primitive::<Decimal128Type>().value(0), 1_500);
                assert_eq!(min.as_primitive::<Decimal128Type>().value(0), -2_000);
            } else {
                assert_eq!(data_type, timestamp_type);
                assert_eq!(max.as_primitive::<TimestampMicrosecondType>().value(0), 9);
                assert_eq!(min.as_primitive::<TimestampMicrosecondType>().value(0), 1);
            }
        }
    }
}
