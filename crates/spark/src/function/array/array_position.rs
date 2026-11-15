// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::float_normalize::normalize_array;
use arrow::array::{
    Array, ArrayRef, AsArray, Int64Array, OffsetSizeTrait, make_comparator,
};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Spark-compatible `array_position` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_position>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns the 1-based position of the first occurrence of the element in the \
        array, as a `BIGINT`. The DataFusion built-in returns a `UInt64`, and returns NULL when \
        the element is not found, where this returns 0. The result is NULL only when the array \
        or the element is NULL: a NULL element returns NULL even if the array contains NULL, \
        while in the DataFusion built-in a NULL element matches the first NULL in the array. \
        NULL elements inside the array never match. For floating point values, also inside \
        nested arrays and structs, NaN is equal to NaN and -0.0 is equal to 0.0. As in Spark, \
        the array element and the element are cast to their tightest common type: numbers \
        widen, an integer widens to a decimal only if the decimal can hold all its values, and \
        a date widens to a timestamp. Other combinations are an error, for example a string \
        and a number, two different decimals, or structs whose field names or field types \
        differ, as in Spark with ANSI mode off (with ANSI mode on, Spark widens struct field \
        types). A NULL-typed array and map elements are also an error.",
    syntax_example = "array_position(array, element)",
    sql_example = r#"```sql
> SELECT array_position(array(3, 2, 1), 1);
3
> SELECT array_position(array(3, 2, 1), 4);
0
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL."),
    argument(
        name = "element",
        description = "Element to search for. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayPosition {
    signature: Signature,
}

impl Default for SparkArrayPosition {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayPosition {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayPosition {
    fn name(&self) -> &str {
        "array_position"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int64)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [array, element] = take_function_args(self.name(), arg_types)?;
        coerce_array_and_element(self.name(), array, element)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let is_scalar = args
            .args
            .iter()
            .all(|a| matches!(a, ColumnarValue::Scalar(_)));
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let [array, element] = take_function_args(self.name(), &arrays)?;

        let result = match array.data_type() {
            DataType::List(_) => position::<i32>(array, element)?,
            DataType::LargeList(_) => position::<i64>(array, element)?,
            other => return exec_err!("array_position does not support type {other}"),
        };

        if is_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn position<O: OffsetSizeTrait>(
    array: &ArrayRef,
    element: &ArrayRef,
) -> Result<ArrayRef> {
    let list = array.as_list::<O>();
    let offsets = list.offsets();
    // Only the values that this (possibly sliced) list refers to are normalized.
    let base = offsets[0].as_usize();
    let len = offsets[offsets.len() - 1].as_usize() - base;
    // Canonicalize NaN and -0.0 at any depth so that the comparator follows Spark's float
    // equality. The comparator treats NULLs nested inside the compared values as equal.
    let values = normalize_array(&list.values().slice(base, len))?;
    let element = normalize_array(element)?;
    let nulls = NullBuffer::union(list.nulls(), element.nulls());
    let cmp = make_comparator(values.as_ref(), element.as_ref(), SortOptions::default())?;

    let result: Vec<i64> = offsets
        .windows(2)
        .enumerate()
        .map(|(row, w)| {
            if nulls.as_ref().is_some_and(|n| n.is_null(row)) {
                return 0;
            }
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            (start..end)
                .find(|&i| values.is_valid(i) && cmp(i, row).is_eq())
                .map_or(0, |i| (i - start + 1) as i64)
        })
        .collect();

    Ok(Arc::new(Int64Array::new(ScalarBuffer::from(result), nulls)))
}

/// Casts the array element and the element to their tightest common type, and rejects the
/// argument types that Spark's `ArrayPosition.checkInputDataTypes` rejects.
fn coerce_array_and_element(
    name: &str,
    array: &DataType,
    element: &DataType,
) -> Result<Vec<DataType>> {
    if array.is_null() {
        return null_type_err(name);
    }
    let (item, large) = match array {
        DataType::List(field) | DataType::FixedSizeList(field, _) => (field, false),
        DataType::LargeList(field) => (field, true),
        other => {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `{name}` \
                 requires the \"ARRAY\" type, however it has the type {other}"
            );
        }
    };
    let Some(common) = tightest_common_type(item.data_type(), element) else {
        return plan_err!(
            "[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES] Input to `{name}` should have \
             been \"ARRAY\" followed by a value with same element type, but it's \
             [{array}, {element}]"
        );
    };
    if common.is_null() {
        return null_type_err(name);
    }
    if contains_map(&common) {
        return plan_err!(
            "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{name}` does not support \
             ordering on type {common}"
        );
    }
    let item = Arc::new(Field::new(item.name(), common.clone(), true));
    let array = if large {
        DataType::LargeList(item)
    } else {
        DataType::List(item)
    };
    Ok(vec![array, common])
}

fn null_type_err<T>(name: &str) -> Result<T> {
    plan_err!(
        "[DATATYPE_MISMATCH.NULL_TYPE] Null typed values cannot be used as arguments of \
         `{name}`"
    )
}

/// Spark's `TypeCoercion.findTightestCommonType`, on the arrow types that Spark's types
/// map to.
fn tightest_common_type(left: &DataType, right: &DataType) -> Option<DataType> {
    use DataType::*;
    if left == right {
        return Some(left.clone());
    }
    if let (Some(l), Some(r)) = (numeric_precedence(left), numeric_precedence(right)) {
        return Some(if l >= r { left.clone() } else { right.clone() });
    }
    match (left, right) {
        (Null, other) | (other, Null) => Some(other.clone()),
        // Spark has a single string type and a single binary type.
        (l, r) if is_string(l) && is_string(r) => Some(l.clone()),
        (l, r) if is_binary(l) && is_binary(r) => Some(l.clone()),
        // The decimal must be at least as wide as the integral type's DECIMAL(n, 0).
        (int, Decimal128(p, s)) | (Decimal128(p, s), int) => {
            let digits = integral_digits(int)?;
            (*p as i16 - *s as i16 >= digits).then_some(Decimal128(*p, *s))
        }
        (Date32, ts @ Timestamp(_, _)) | (ts @ Timestamp(_, _), Date32) => {
            Some(ts.clone())
        }
        // TIMESTAMP_NTZ and TIMESTAMP widen to TIMESTAMP.
        (Timestamp(lu, ltz), Timestamp(ru, rtz)) => Some(Timestamp(
            finer_unit(*lu, *ru),
            ltz.clone().or_else(|| rtz.clone()),
        )),
        (
            List(lf) | LargeList(lf) | FixedSizeList(lf, _),
            List(rf) | LargeList(rf) | FixedSizeList(rf, _),
        ) => {
            let item = tightest_common_type(lf.data_type(), rf.data_type())?;
            let field = Arc::new(Field::new(lf.name(), item, true));
            Some(
                if matches!(left, LargeList(_)) || matches!(right, LargeList(_)) {
                    LargeList(field)
                } else {
                    List(field)
                },
            )
        }
        // The fields must have the same names and types, ignoring nullability, in the same
        // order. This follows Spark with ANSI mode off; with ANSI mode on, Spark also widens
        // the field types. Spark compares the names ignoring case, but DataFusion cannot cast
        // between structs whose field names differ only in case, so they must match exactly.
        (Struct(lf), Struct(rf)) if lf.len() == rf.len() => {
            let fields = lf
                .iter()
                .zip(rf.iter())
                .map(|(a, b)| {
                    if a.name() != b.name() || !same_type(a.data_type(), b.data_type()) {
                        return None;
                    }
                    let data_type = tightest_common_type(a.data_type(), b.data_type())?;
                    Some(Field::new(
                        a.name(),
                        data_type,
                        a.is_nullable() || b.is_nullable(),
                    ))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Struct(Fields::from(fields)))
        }
        (Map(lf, sorted), Map(rf, _)) => {
            let entries = tightest_common_type(lf.data_type(), rf.data_type())?;
            Some(Map(
                Arc::new(Field::new(lf.name(), entries, false)),
                *sorted,
            ))
        }
        _ => None,
    }
}

/// Whether two types are equal when the nullability of nested fields is ignored, like
/// Spark's `DataTypeUtils.sameType`.
fn same_type(left: &DataType, right: &DataType) -> bool {
    use DataType::*;
    match (left, right) {
        (List(l), List(r)) | (LargeList(l), LargeList(r)) | (Map(l, _), Map(r, _)) => {
            same_type(l.data_type(), r.data_type())
        }
        (FixedSizeList(l, ln), FixedSizeList(r, rn)) => {
            ln == rn && same_type(l.data_type(), r.data_type())
        }
        (Struct(l), Struct(r)) => {
            l.len() == r.len()
                && l.iter().zip(r.iter()).all(|(a, b)| {
                    a.name() == b.name() && same_type(a.data_type(), b.data_type())
                })
        }
        _ => left == right,
    }
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn is_binary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    )
}

/// The precision of Spark's `DecimalType.forType` for an integral type.
fn integral_digits(data_type: &DataType) -> Option<i16> {
    match data_type {
        DataType::Int8 => Some(3),
        DataType::Int16 => Some(5),
        DataType::Int32 => Some(10),
        DataType::Int64 => Some(20),
        _ => None,
    }
}

/// Spark's `numericPrecedence`, which leaves out decimals.
fn numeric_precedence(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int8 => Some(0),
        DataType::Int16 => Some(1),
        DataType::Int32 => Some(2),
        DataType::Int64 => Some(3),
        DataType::Float32 => Some(4),
        DataType::Float64 => Some(5),
        _ => None,
    }
}

fn finer_unit(left: TimeUnit, right: TimeUnit) -> TimeUnit {
    let rank = |unit: TimeUnit| match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    };
    if rank(left) >= rank(right) {
        left
    } else {
        right
    }
}

/// Spark cannot order maps, so it rejects them at any depth of the element type.
fn contains_map(data_type: &DataType) -> bool {
    match data_type {
        DataType::Map(_, _) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => contains_map(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| contains_map(field.data_type()))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float64Array, Int32Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::Int32Type;

    #[test]
    fn sliced_list() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(9), Some(9)]),
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(3), Some(4)]),
        ]);
        let array: ArrayRef = Arc::new(list.slice(1, 2));
        let element: ArrayRef = Arc::new(Int32Array::from(vec![3, 4]));
        let result = position::<i32>(&array, &element).unwrap();
        assert_eq!(result.as_ref(), &Int64Array::from(vec![3, 2]));
    }

    #[test]
    fn fixed_size_list_of_floats() {
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let fsl = |values: Vec<f64>| -> ArrayRef {
            Arc::new(
                FixedSizeListArray::try_new(
                    Arc::clone(&item),
                    1,
                    Arc::new(Float64Array::from(values)),
                    None,
                )
                .unwrap(),
            )
        };
        let values = fsl(vec![1.0, -0.0, f64::NAN]);
        let field = Arc::new(Field::new("item", values.data_type().clone(), true));
        let array: ArrayRef = Arc::new(
            ListArray::try_new(
                field,
                OffsetBuffer::new(vec![0, 3, 6].into()),
                Arc::new(arrow::compute::concat(&[&values, &values]).unwrap()),
                None,
            )
            .unwrap(),
        );
        let element = fsl(vec![0.0, -f64::NAN]);
        let result = position::<i32>(&array, &element).unwrap();
        assert_eq!(result.as_ref(), &Int64Array::from(vec![2, 3]));
    }
}
