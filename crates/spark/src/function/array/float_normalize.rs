// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Float normalization shared by the array functions that compare elements.
//!
//! Spark compares floating point values with `-0.0 == 0.0` and `NaN == NaN`, while arrow
//! compares them by their bits or by the IEEE total order. Rewriting `-0.0` to `0.0` and
//! every NaN to the canonical NaN, at any depth of nested types, makes those comparisons
//! follow Spark's float equality.

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, GenericListArray, MapArray,
    OffsetSizeTrait, StructArray,
};
use arrow::datatypes::{DataType, FieldRef, Float32Type, Float64Type};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use std::sync::Arc;

/// Normalizes the floats in a [`ColumnarValue`], see [`normalize_array`].
pub(super) fn normalize_floats(value: &ColumnarValue) -> Result<ColumnarValue> {
    if !contains_float(&value.data_type()) {
        return Ok(value.clone());
    }
    match value {
        ColumnarValue::Array(array) => Ok(ColumnarValue::Array(normalize_array(array)?)),
        ColumnarValue::Scalar(scalar) => {
            let array = normalize_array(&scalar.to_array()?)?;
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &array, 0,
            )?))
        }
    }
}

/// Returns whether values of this type are, or hold at any depth, floating point numbers.
pub(super) fn contains_float(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_float(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| contains_float(field.data_type()))
        }
        _ => false,
    }
}

/// Rewrites `-0.0` to `0.0` and every NaN to the canonical NaN at any depth of list, large
/// list, fixed-size list, struct and map. Children without floats are reused as they are.
pub(super) fn normalize_array(array: &ArrayRef) -> Result<ArrayRef> {
    if !contains_float(array.data_type()) {
        return Ok(Arc::clone(array));
    }
    Ok(match array.data_type() {
        DataType::Float32 => Arc::new(
            array
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|v| {
                    if v.is_nan() {
                        f32::NAN
                    } else if v == 0.0 {
                        0.0
                    } else {
                        v
                    }
                }),
        ),
        DataType::Float64 => Arc::new(
            array
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|v| {
                    if v.is_nan() {
                        f64::NAN
                    } else if v == 0.0 {
                        0.0
                    } else {
                        v
                    }
                }),
        ),
        DataType::List(field) => normalize_list(array.as_list::<i32>(), field)?,
        DataType::LargeList(field) => normalize_list(array.as_list::<i64>(), field)?,
        DataType::FixedSizeList(field, size) => {
            let list = array.as_fixed_size_list();
            // The length is passed explicitly: a zero-width list has no values to infer it from.
            Arc::new(FixedSizeListArray::try_new_with_length(
                Arc::clone(field),
                *size,
                normalize_array(list.values())?,
                list.nulls().cloned(),
                list.len(),
            )?)
        }
        DataType::Struct(fields) => {
            let columns = array
                .as_struct()
                .columns()
                .iter()
                .map(normalize_array)
                .collect::<Result<Vec<_>>>()?;
            Arc::new(StructArray::try_new(
                fields.clone(),
                columns,
                array.nulls().cloned(),
            )?)
        }
        DataType::Map(field, ordered) => {
            let map = array.as_map();
            let entries: ArrayRef = Arc::new(map.entries().clone());
            let entries = normalize_array(&entries)?;
            Arc::new(MapArray::try_new(
                Arc::clone(field),
                map.offsets().clone(),
                entries.as_struct().clone(),
                map.nulls().cloned(),
                *ordered,
            )?)
        }
        _ => Arc::clone(array),
    })
}

fn normalize_list<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    field: &FieldRef,
) -> Result<ArrayRef> {
    Ok(Arc::new(GenericListArray::<O>::try_new(
        Arc::clone(field),
        list.offsets().clone(),
        normalize_array(list.values())?,
        list.nulls().cloned(),
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Float32Array, Float64Array, Int32Array, Int32Builder, ListArray, ListBuilder,
        MapBuilder,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::{Field, Fields, Int32Type};

    fn negative_nan() -> f64 {
        f64::from_bits(f64::NAN.to_bits() | (1 << 63))
    }

    #[test]
    fn normalize_array_canonicalizes_floats() {
        let array: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(-0.0),
            Some(negative_nan()),
            Some(1.5),
            None,
        ]));
        let normalized = normalize_array(&array).unwrap();
        let normalized = normalized.as_primitive::<Float64Type>();
        assert_eq!(normalized.value(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(normalized.value(1).to_bits(), f64::NAN.to_bits());
        assert_eq!(normalized.value(2), 1.5);
        assert!(normalized.is_null(3));

        let array: ArrayRef = Arc::new(Float32Array::from(vec![-0.0_f32, f32::NAN]));
        let normalized = normalize_array(&array).unwrap();
        let normalized = normalized.as_primitive::<Float32Type>();
        assert_eq!(normalized.value(0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(normalized.value(1).to_bits(), f32::NAN.to_bits());
    }

    #[test]
    fn normalize_array_keeps_the_shape_of_nested_values() {
        let fields = Fields::from(vec![
            Field::new("a", DataType::Float64, true),
            Field::new("b", DataType::Int32, true),
        ]);
        let item = Arc::new(Field::new("item", DataType::Struct(fields.clone()), true));
        let structs = StructArray::try_new(
            fields,
            vec![
                Arc::new(Float64Array::from(vec![-0.0, 2.0, 3.0])) as ArrayRef,
                Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            ],
            Some(NullBuffer::from(vec![true, false, true])),
        )
        .unwrap();
        let list = ListArray::try_new(
            item,
            OffsetBuffer::new(vec![0, 2, 3].into()),
            Arc::new(structs),
            Some(NullBuffer::from(vec![true, false])),
        )
        .unwrap();
        let array: ArrayRef = Arc::new(list);
        assert!(contains_float(array.data_type()));

        let normalized = normalize_array(&array).unwrap();
        assert_eq!(normalized.data_type(), array.data_type());
        assert_eq!(normalized.nulls(), array.nulls());
        let original = array.as_list::<i32>();
        let list = normalized.as_list::<i32>();
        assert_eq!(list.offsets(), original.offsets());
        let structs = list.values().as_struct();
        assert_eq!(structs.nulls(), original.values().nulls());
        let a = structs.column(0).as_primitive::<Float64Type>();
        assert_eq!(a.value(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(a.value(1), 2.0);
        assert_eq!(structs.column(1).as_primitive::<Int32Type>().value(2), 3);
    }

    #[test]
    fn normalize_array_descends_into_maps() {
        let mut builder = MapBuilder::new(
            None,
            Int32Builder::new(),
            arrow::array::Float64Builder::new(),
        );
        builder.keys().append_value(1);
        builder.values().append_value(-0.0);
        builder.keys().append_value(2);
        builder.values().append_value(negative_nan());
        builder.append(true).unwrap();
        builder.append(false).unwrap();
        let array: ArrayRef = Arc::new(builder.finish());
        assert!(contains_float(array.data_type()));

        let normalized = normalize_array(&array).unwrap();
        assert_eq!(normalized.data_type(), array.data_type());
        assert_eq!(normalized.nulls(), array.nulls());
        let map = normalized.as_map();
        let values = map.values().as_primitive::<Float64Type>();
        assert_eq!(values.value(0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(values.value(1).to_bits(), f64::NAN.to_bits());
        assert_eq!(map.keys().as_primitive::<Int32Type>().value(1), 2);
    }

    #[test]
    fn normalize_floats_leaves_other_types_untouched() {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.values().append_value(1);
        builder.append(true);
        let array: ArrayRef = Arc::new(builder.finish());
        assert!(!contains_float(array.data_type()));
        let value = ColumnarValue::Array(Arc::clone(&array));
        match normalize_floats(&value).unwrap() {
            ColumnarValue::Array(result) => assert!(Arc::ptr_eq(&result, &array)),
            ColumnarValue::Scalar(_) => panic!("expected an array"),
        }
    }

    #[test]
    fn normalize_array_reuses_children_without_floats() {
        let mut builder = ListBuilder::new(Int32Builder::new());
        builder.values().append_value(1);
        builder.append(true);
        let ints: ArrayRef = Arc::new(builder.finish());
        let structs = StructArray::try_new(
            Fields::from(vec![
                Field::new("a", DataType::Float64, true),
                Field::new("b", ints.data_type().clone(), true),
            ]),
            vec![
                Arc::new(Float64Array::from(vec![-0.0])) as ArrayRef,
                Arc::clone(&ints),
            ],
            None,
        )
        .unwrap();

        let normalized = normalize_array(&(Arc::new(structs) as ArrayRef)).unwrap();
        assert!(Arc::ptr_eq(normalized.as_struct().column(1), &ints));
    }

    #[test]
    fn zero_width_fixed_size_list_nested_in_a_list_is_not_rebuilt() {
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let inner = Arc::new(Field::new(
            "inner",
            DataType::FixedSizeList(Arc::clone(&item), 0),
            true,
        ));
        let zero_width = FixedSizeListArray::try_new_with_length(
            Arc::clone(&item),
            0,
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            None,
            1,
        )
        .unwrap();
        let list = ListArray::try_new(
            inner,
            OffsetBuffer::new(vec![0, 1].into()),
            Arc::new(zero_width),
            None,
        )
        .unwrap();
        let array: ArrayRef = Arc::new(list);

        let normalized = normalize_array(&array).unwrap();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized.as_list::<i32>().values().len(), 1);
    }
}
