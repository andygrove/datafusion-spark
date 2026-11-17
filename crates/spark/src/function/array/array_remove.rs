// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::float_normalize::normalize_array;
use crate::function::type_coercion::{
    is_orderable, spark_sql_type, struct_field_types_match, tightest_common_type,
};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, AsArray, BooleanArray, BooleanBufferBuilder,
    GenericListArray, OffsetSizeTrait, Scalar, make_comparator,
};
use arrow::buffer::{BooleanBuffer, NullBuffer, OffsetBuffer};
use arrow::compute::kernels::cmp::eq;
use arrow::compute::{SortOptions, filter};
use arrow::datatypes::{DataType, Field, Float32Type, Float64Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;
use num_traits::Float;
use std::sync::Arc;

/// Spark-compatible `array_remove` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_remove>
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Removes all elements that are equal to the element from the array, while \
        the DataFusion built-in removes only the first one. The result is NULL when the array \
        or the element is NULL. NULL elements in the array are always kept. Corresponding NULLs nested inside compared arrays or structs are equal. \
        For floating point values, also inside nested arrays and structs, NaN is equal to NaN \
        and -0.0 is equal to 0.0; the elements that are kept keep their original values. As \
        in Spark, the array element and the element are cast to their tightest common type, \
        and the result is an array of that type: numbers widen, an integer widens to a \
        decimal only if the decimal can hold all its values, and a date widens to a \
        timestamp. Other combinations are an error, for example a string and a number, two \
        different decimals, or structs whose field names or field types differ, as in Spark \
        with ANSI mode off (with ANSI mode on, Spark widens struct field types). Map and \
        INTERVAL elements, at any depth, are also an error.",
    syntax_example = "array_remove(array, element)",
    sql_example = r#"```sql
> SELECT array_remove(array(1, 2, 3, NULL, 3), 3);
[1, 2, NULL]
```"#,
    argument(name = "array", description = "Array expression. NULL returns NULL."),
    argument(
        name = "element",
        description = "Element to remove. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayRemove {
    signature: Signature,
}

impl Default for SparkArrayRemove {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayRemove {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayRemove {
    fn name(&self) -> &str {
        "array_remove"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [array, _] = take_function_args(self.name(), arg_types)?;
        Ok(array.clone())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [array, element] = take_function_args(self.name(), arg_types)?;
        coerce_array_and_element(self.name(), array, element)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [array, element] = take_function_args(self.name(), &args.args)?;
        let is_scalar = matches!(
            (array, element),
            (ColumnarValue::Scalar(_), ColumnarValue::Scalar(_))
        );
        let rows = if is_scalar { 1 } else { args.number_rows };
        let array = array.to_array(rows)?;
        // A constant element is compared as a single value against every row.
        let (element, constant) = match element {
            ColumnarValue::Scalar(scalar) => (scalar.to_array()?, true),
            ColumnarValue::Array(element) => (Arc::clone(element), false),
        };

        let result = match array.data_type() {
            DataType::List(_) => remove::<i32>(&array, &element, constant)?,
            DataType::LargeList(_) => remove::<i64>(&array, &element, constant)?,
            other => return exec_err!("array_remove does not support type {other}"),
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

/// Removes from each row of `array` the non-null elements equal to the row's element, which
/// is `element[0]` for every row if `constant`, and `element[row]` otherwise. A row is NULL
/// when the array or the element is NULL.
fn remove<O: OffsetSizeTrait>(
    array: &ArrayRef,
    element: &ArrayRef,
    constant: bool,
) -> Result<ArrayRef> {
    let list = array.as_list::<O>();
    let offsets = list.offsets();
    // Only the values that this (possibly sliced) list refers to are looked at.
    let base = offsets[0].as_usize();
    let len = offsets[offsets.len() - 1].as_usize() - base;
    let values = list.values().slice(base, len);

    // Logical nulls, so that a NULL-typed element, which has no null buffer, is NULL.
    let element_nulls = element.logical_nulls();
    let row_nulls = if constant {
        if element_nulls.as_ref().is_some_and(|n| n.is_null(0)) {
            Some(NullBuffer::new_null(list.len()))
        } else {
            list.nulls().cloned()
        }
    } else {
        NullBuffer::union(list.nulls(), element_nulls.as_ref())
    };
    let row_is_valid = |row: usize| row_nulls.as_ref().is_none_or(|n| n.is_valid(row));

    // When every element is NULL there is nothing to compare: NULL elements are kept and a
    // NULL element makes every row NULL. This also covers the NULL type.
    let any_comparison =
        values.logical_null_count() < values.len() && (0..list.len()).any(row_is_valid);
    let value_nulls = values.logical_nulls();
    let mut new_offsets = Vec::with_capacity(list.len() + 1);
    new_offsets.push(O::usize_as(0));
    let mut kept = 0usize;

    // A caller that skips coercion, such as Comet, can pass an element of another type. The
    // bitmap path would panic on its downcast, so such an element takes the comparator path,
    // which returns an error.
    let keep = if any_comparison
        && constant
        && !values.data_type().is_nested()
        && element.data_type() == values.data_type()
    {
        // A constant element that is not nested is compared with every value at once, and
        // the elements to keep are found with bitmap operations.
        let equal = match values.data_type() {
            DataType::Float32 => float_equal::<Float32Type>(&values, element),
            DataType::Float64 => float_equal::<Float64Type>(&values, element),
            _ => eq(&values, &Scalar::new(Arc::clone(element)))?
                .values()
                .clone(),
        };
        let mut keep = !&equal;
        if let Some(nulls) = &value_nulls {
            keep = &keep | &!nulls.inner();
        }
        if row_nulls.as_ref().is_some_and(|n| n.null_count() > 0) {
            // The elements of a NULL row are dropped.
            let mut in_valid_row = BooleanBufferBuilder::new(len);
            for (row, w) in offsets.windows(2).enumerate() {
                in_valid_row.append_n((w[1] - w[0]).as_usize(), row_is_valid(row));
            }
            keep = &keep & &in_valid_row.finish();
        }
        for w in offsets.windows(2) {
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            kept += keep.slice(start, end - start).count_set_bits();
            new_offsets.push(O::usize_as(kept));
        }
        keep
    } else {
        // Equality is computed on copies with canonical NaN and 0.0 at any depth, so that it
        // follows Spark's float equality, while the kept elements are taken from the
        // original values. The comparator treats NULLs nested inside the compared values as
        // equal.
        let mut is_equal: Box<dyn Fn(usize, usize) -> bool> = Box::new(|_, _| false);
        if any_comparison {
            let normalized = normalize_array(&values)?;
            let needle = normalize_array(element)?;
            let cmp = make_comparator(
                normalized.as_ref(),
                needle.as_ref(),
                SortOptions::default(),
            )?;
            is_equal =
                Box::new(move |i, row| cmp(i, if constant { 0 } else { row }).is_eq());
        }
        let mut keep = BooleanBufferBuilder::new(len);
        for (row, w) in offsets.windows(2).enumerate() {
            let (start, end) = (w[0].as_usize() - base, w[1].as_usize() - base);
            if row_is_valid(row) {
                for i in start..end {
                    let keep_it = value_nulls.as_ref().is_some_and(|n| n.is_null(i))
                        || !is_equal(i, row);
                    keep.append(keep_it);
                    kept += usize::from(keep_it);
                }
            } else {
                keep.append_n(end - start, false);
            }
            new_offsets.push(O::usize_as(kept));
        }
        keep.finish()
    };
    let kept_values = if kept == len {
        values
    } else {
        filter(&values, &BooleanArray::new(keep, None))?
    };

    let (DataType::List(field) | DataType::LargeList(field)) = list.data_type() else {
        unreachable!("remove is only called with lists");
    };
    Ok(Arc::new(GenericListArray::<O>::try_new(
        Arc::clone(field),
        OffsetBuffer::new(new_offsets.into()),
        kept_values,
        row_nulls,
    )?))
}

/// For each value, whether it equals the one-row `element` under Spark's float equality,
/// where NaN equals NaN and `-0.0` equals `0.0`. The bits of NULL values are arbitrary.
fn float_equal<T>(values: &ArrayRef, element: &ArrayRef) -> BooleanBuffer
where
    T: ArrowPrimitiveType,
    T::Native: Float,
{
    let values = values.as_primitive::<T>().values();
    let needle = element.as_primitive::<T>().value(0);
    if needle.is_nan() {
        BooleanBuffer::collect_bool(values.len(), |i| values[i].is_nan())
    } else {
        // IEEE 754 `==` already treats `-0.0` as equal to `0.0`.
        BooleanBuffer::collect_bool(values.len(), |i| values[i] == needle)
    }
}

/// Casts the array element and the element to their tightest common type, and rejects the
/// argument types that Spark's `ArrayRemove.checkInputDataTypes` rejects.
///
/// Unlike `array_position`, Spark accepts a NULL-typed common element type here (for
/// example `array_remove(array(), NULL)`, which returns NULL), and reports an argument that
/// is not an array, including an untyped NULL, as `ARRAY_FUNCTION_DIFF_TYPES`.
fn coerce_array_and_element(
    name: &str,
    array: &DataType,
    element: &DataType,
) -> Result<Vec<DataType>> {
    let diff_types_err = || {
        plan_err!(
            "[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES] Input to `{name}` should have \
             been \"ARRAY\" followed by a value with same element type, but it's \
             [\"{}\", \"{}\"]. SQLSTATE: 42K09",
            spark_sql_type(array),
            spark_sql_type(element)
        )
    };
    let (item, large) = match array {
        DataType::List(field) | DataType::FixedSizeList(field, _) => (field, false),
        DataType::LargeList(field) => (field, true),
        _ => return diff_types_err(),
    };
    // Spark's `ArrayRemove` calls the non-ANSI `TypeCoercion.findTightestCommonType`.
    let common = tightest_common_type(item.data_type(), element, false)
        .filter(|_| struct_field_types_match(item.data_type(), element));
    let Some(common) = common else {
        return diff_types_err();
    };
    if !is_orderable(&common) {
        return plan_err!(
            "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{name}` does not support \
             ordering on type \"{}\". SQLSTATE: 42K09",
            spark_sql_type(&common)
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array, Int32Array, ListArray, StructArray};
    use arrow::datatypes::{Fields, Int32Type};

    fn negative_nan() -> f64 {
        f64::from_bits(f64::NAN.to_bits() | (1 << 63))
    }

    fn invoke(
        array: ColumnarValue,
        element: ColumnarValue,
        rows: usize,
    ) -> ColumnarValue {
        let return_field = Arc::new(Field::new("result", array.data_type(), true));
        SparkArrayRemove::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![array, element],
                arg_fields: vec![],
                number_rows: rows,
                return_field,
                config_options: Default::default(),
            })
            .unwrap()
    }

    fn float_list(rows: &[Option<Vec<Option<f64>>>]) -> ArrayRef {
        Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(
            rows.to_vec(),
        ))
    }

    /// The rows of a list of doubles, as bits so that the zeros and NaNs can be told apart.
    fn bits(array: &ArrayRef) -> Vec<Option<Vec<Option<u64>>>> {
        array
            .as_list::<i32>()
            .iter()
            .map(|row| {
                row.map(|row| {
                    row.as_primitive::<Float64Type>()
                        .iter()
                        .map(|v| v.map(f64::to_bits))
                        .collect()
                })
            })
            .collect()
    }

    /// Spark's result for one row: drop the non-null elements equal to the element.
    fn spark_remove(row: &[Option<f64>], element: f64) -> Vec<Option<u64>> {
        let equal = |a: f64| (a.is_nan() && element.is_nan()) || a == element;
        row.iter()
            .filter(|v| v.is_none_or(|v| !equal(v)))
            .map(|v| v.map(f64::to_bits))
            .collect()
    }

    const EDGE_VALUES: [Option<f64>; 7] = [
        Some(0.0),
        Some(-0.0),
        Some(f64::NAN),
        None,
        Some(1.0),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ];

    /// Every edge value removed from an array of all edge values, both as a column and as a
    /// constant: each zero and each NaN meets every other, and the kept elements keep their
    /// bits.
    #[test]
    fn floats_follow_spark_equality() {
        let mut elements = EDGE_VALUES.to_vec();
        elements.push(Some(negative_nan()));
        let removable: Vec<f64> = elements.iter().flatten().copied().collect();
        let rows: Vec<_> = removable.iter().map(|_| Some(elements.clone())).collect();
        let expected: Vec<_> = removable
            .iter()
            .map(|&v| Some(spark_remove(&elements, v)))
            .collect();

        let result = invoke(
            ColumnarValue::Array(float_list(&rows)),
            ColumnarValue::Array(Arc::new(Float64Array::from(removable.clone()))),
            rows.len(),
        )
        .into_array(rows.len())
        .unwrap();
        assert_eq!(bits(&result), expected);

        for (value, expected) in removable.iter().zip(expected) {
            let result = invoke(
                ColumnarValue::Array(float_list(&rows[..1])),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(*value))),
                1,
            )
            .into_array(1)
            .unwrap();
            assert_eq!(bits(&result), vec![expected], "remove {value}");
        }
    }

    #[test]
    fn nulls() {
        let rows = vec![
            Some(vec![Some(0.0), None, Some(-0.0)]),
            None,
            Some(vec![Some(1.0)]),
        ];
        let element = Float64Array::from(vec![Some(0.0), Some(0.0), None]);
        let result = invoke(
            ColumnarValue::Array(float_list(&rows)),
            ColumnarValue::Array(Arc::new(element)),
            3,
        )
        .into_array(3)
        .unwrap();
        // The NULL element stays; a NULL array or a NULL element gives a NULL row.
        assert_eq!(bits(&result), vec![Some(vec![None]), None, None]);

        let result = invoke(
            ColumnarValue::Array(float_list(&rows)),
            ColumnarValue::Scalar(ScalarValue::Float64(None)),
            3,
        )
        .into_array(3)
        .unwrap();
        assert_eq!(bits(&result), vec![None, None, None]);
    }

    /// The offsets of a sliced list do not start at 0, and the elements of NULL rows are not
    /// carried into the result.
    #[test]
    fn sliced_list() {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(9), Some(9)]),
            Some(vec![Some(1), Some(2), Some(1)]),
            None,
            Some(vec![Some(3), None, Some(4)]),
        ]);
        let array: ArrayRef = Arc::new(list.slice(1, 3));
        for element in [
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![1, 1, 1]))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
        ] {
            let result = invoke(ColumnarValue::Array(Arc::clone(&array)), element, 3)
                .into_array(3)
                .unwrap();
            let expected = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(2)]),
                None,
                Some(vec![Some(3), None, Some(4)]),
            ]);
            assert_eq!(result.as_list::<i32>(), &expected);
            assert_eq!(result.as_list::<i32>().values().len(), 4);
        }
    }

    #[test]
    fn large_list_and_scalars() {
        let array = ScalarValue::LargeList(Arc::new(
            arrow::array::LargeListArray::from_iter_primitive::<Float32Type, _, _>(vec![
                Some(vec![Some(-0.0f32), Some(f32::NAN), Some(1.0)]),
            ]),
        ));
        let result = invoke(
            ColumnarValue::Scalar(array),
            ColumnarValue::Scalar(ScalarValue::Float32(Some(0.0))),
            4,
        );
        let ColumnarValue::Scalar(ScalarValue::LargeList(result)) = result else {
            panic!("expected a large list scalar");
        };
        let values = result.value(0);
        let values: &Float32Array = values.as_primitive();
        assert_eq!(values.len(), 2);
        assert!(values.value(0).is_nan());
        assert_eq!(values.value(1), 1.0);
    }

    /// Struct elements compare their float fields with Spark's equality, and keep their bits.
    #[test]
    fn nested_elements() {
        let fields = Fields::from(vec![Field::new("x", DataType::Float64, true)]);
        let structs = |values: Vec<f64>| -> ArrayRef {
            Arc::new(StructArray::new(
                fields.clone(),
                vec![Arc::new(Float64Array::from(values))],
                None,
            ))
        };
        let item = Field::new("item", DataType::Struct(fields.clone()), true);
        let array: ArrayRef = Arc::new(ListArray::new(
            Arc::new(item),
            OffsetBuffer::from_lengths([3]),
            structs(vec![-0.0, negative_nan(), 1.0]),
            None,
        ));
        for (value, expected) in [
            (0.0, vec![negative_nan(), 1.0]),
            (f64::NAN, vec![-0.0, 1.0]),
        ] {
            for element in [
                ColumnarValue::Array(structs(vec![value])),
                ColumnarValue::Scalar(
                    ScalarValue::try_from_array(&structs(vec![value]), 0).unwrap(),
                ),
            ] {
                let result = invoke(ColumnarValue::Array(Arc::clone(&array)), element, 1)
                    .into_array(1)
                    .unwrap();
                let row = result.as_list::<i32>().value(0);
                let x = row.as_struct().column(0).as_primitive::<Float64Type>();
                let x: Vec<u64> = x.values().iter().map(|v| v.to_bits()).collect();
                let expected: Vec<u64> = expected.iter().map(|v| v.to_bits()).collect();
                assert_eq!(x, expected, "remove {value}");
            }
        }
    }

    /// Without coercion, an element whose type differs from the array's element type is an
    /// error, whether it is a constant or a column.
    #[test]
    fn mismatched_element_type_is_an_error() {
        let array = float_list(&[Some(vec![Some(1.0), Some(2.0)])]);
        let ints: ArrayRef =
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(1), Some(2)]),
            ]));
        for (array, element) in [
            (&array, ScalarValue::Float32(Some(1.0))),
            (&array, ScalarValue::Int64(Some(1))),
            (&ints, ScalarValue::Int64(Some(1))),
        ] {
            for constant in [true, false] {
                let element = element.to_array().unwrap();
                let err = remove::<i32>(array, &element, constant)
                    .unwrap_err()
                    .to_string();
                assert!(
                    err.contains("Can't compare arrays of different types"),
                    "{err}"
                );
            }
        }
    }

    #[test]
    fn coerce_types_rules() {
        let udf = SparkArrayRemove::new();
        let list = |t: DataType| DataType::new_list(t, true);
        assert_eq!(
            udf.coerce_types(&[list(DataType::Int32), DataType::Int64])
                .unwrap(),
            vec![list(DataType::Int64), DataType::Int64]
        );
        assert_eq!(
            udf.coerce_types(&[list(DataType::Null), DataType::Null])
                .unwrap(),
            vec![list(DataType::Null), DataType::Null]
        );
        for (array, element) in [
            (DataType::Null, DataType::Int32),
            (DataType::Int32, DataType::Int32),
            (list(DataType::Int32), DataType::Utf8),
            (list(DataType::Int32), DataType::Decimal128(2, 1)),
        ] {
            let err = udf.coerce_types(&[array, element]).unwrap_err().to_string();
            assert!(err.contains("ARRAY_FUNCTION_DIFF_TYPES"), "{err}");
        }
    }

    /// The bitmap path for a constant element that is not nested must agree with the
    /// comparator path, which a column holding the same element takes.
    #[test]
    fn constant_fast_path_matches_comparator_path() {
        let edge = [
            Some(1.0),
            Some(-0.0),
            Some(0.0),
            Some(f64::NAN),
            Some(negative_nan()),
            Some(f64::INFINITY),
            None,
        ];
        // Every edge value in several orders, with NULL rows and empty rows.
        let rows: Vec<Option<Vec<Option<f64>>>> = (0..edge.len())
            .map(|i| Some(edge.iter().cycle().skip(i).take(i + 2).copied().collect()))
            .chain([None, Some(vec![]), Some(vec![None])])
            .collect();
        let list = ListArray::from_iter_primitive::<Float64Type, _, _>(rows.clone());
        let n = rows.len();
        for array in [
            Arc::new(list.clone()) as ArrayRef,
            Arc::new(list.slice(1, n - 2)),
        ] {
            let len = array.len();
            for needle in edge {
                let constant: ArrayRef = Arc::new(Float64Array::from(vec![needle]));
                let column: ArrayRef = Arc::new(Float64Array::from(vec![needle; len]));
                let fast = remove::<i32>(&array, &constant, true).unwrap();
                let generic = remove::<i32>(&array, &column, false).unwrap();
                assert_eq!(
                    format!("{fast:?}"),
                    format!("{generic:?}"),
                    "needle {needle:?}"
                );
            }
        }

        let ints = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), None, Some(2), Some(1)]),
            None,
            Some(vec![Some(2)]),
        ]);
        let ints: ArrayRef = Arc::new(ints);
        let fast = remove::<i32>(
            &ints,
            &(Arc::new(Int32Array::from(vec![1])) as ArrayRef),
            true,
        )
        .unwrap();
        let generic = remove::<i32>(
            &ints,
            &(Arc::new(Int32Array::from(vec![1; 3])) as ArrayRef),
            false,
        )
        .unwrap();
        assert_eq!(format!("{fast:?}"), format!("{generic:?}"));
    }
}
