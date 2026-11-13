// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, BooleanBufferBuilder, FixedSizeListArray,
    GenericListArray, OffsetSizeTrait, StructArray,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{DataType, FieldRef, Float32Type, Float64Type};
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions_nested::array_has::array_has_udf;
use std::sync::Arc;

/// Spark-compatible `array_contains` function.
///
/// Calls DataFusion's `array_has` and then applies Spark's null semantics:
/// - If the result from `array_has` is `true`, return `true`.
/// - If the result is `false` and the input array row contains any null elements,
///   return `null` (because the element might have been the null).
/// - If the result is `false` and the input array row has no null elements,
///   return `false`.
///
/// Spark compares floating point elements with `-0.0 == 0.0` and `NaN == NaN`,
/// while `array_has` compares them by their bits. Both arguments are therefore
/// normalized first, at any depth of list and struct elements.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayContains {
    signature: Signature,
}

impl Default for SparkArrayContains {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayContains {
    pub fn new() -> Self {
        Self {
            signature: Signature::array_and_element(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkArrayContains {
    fn name(&self) -> &str {
        "array_contains"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, mut args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        for arg in &mut args.args {
            *arg = normalize_floats(arg)?;
        }
        let haystack = args.args[0].clone();
        let array_has_result = array_has_udf().invoke_with_args(args)?;

        let result_array = array_has_result.to_array(1)?;
        let patched = apply_spark_null_semantics(result_array.as_boolean(), &haystack)?;
        Ok(ColumnarValue::Array(Arc::new(patched)))
    }
}

/// Rewrites `-0.0` to `0.0` and every NaN to the canonical NaN, so that `array_has`
/// compares floating point values the way Spark does.
fn normalize_floats(value: &ColumnarValue) -> Result<ColumnarValue> {
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
fn contains_float(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => contains_float(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| contains_float(field.data_type()))
        }
        _ => false,
    }
}

fn normalize_array(array: &ArrayRef) -> Result<ArrayRef> {
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

/// For each row where `array_has` returned `false`, set the output to null
/// if that row's input array contains any null elements.
fn apply_spark_null_semantics(
    result: &BooleanArray,
    haystack_arg: &ColumnarValue,
) -> Result<BooleanArray> {
    // happy path
    if haystack_arg.data_type() == DataType::Null || !result.has_false() {
        return Ok(result.clone());
    }

    let haystack = haystack_arg.to_array_of_size(result.len())?;

    let row_has_nulls = compute_row_has_nulls(&haystack)?;

    // A row keeps its validity when result is true OR the row has no nulls.
    let keep_mask = result.values() | &!&row_has_nulls;
    let new_validity = match result.nulls() {
        Some(n) => n.inner() & &keep_mask,
        None => keep_mask,
    };

    Ok(BooleanArray::new(
        result.values().clone(),
        Some(NullBuffer::new(new_validity)),
    ))
}

/// Returns a per-row bitmap where bit i is set if row i's list contains any null element.
fn compute_row_has_nulls(haystack: &dyn Array) -> Result<BooleanBuffer> {
    match haystack.data_type() {
        DataType::List(_) => generic_list_row_has_nulls(haystack.as_list::<i32>()),
        DataType::LargeList(_) => generic_list_row_has_nulls(haystack.as_list::<i64>()),
        DataType::FixedSizeList(_, _) => {
            let list = haystack.as_fixed_size_list();
            let buf = match list.values().nulls() {
                Some(nulls) => {
                    let validity = nulls.inner();
                    let vl = list.value_length() as usize;
                    let mut builder = BooleanBufferBuilder::new(list.len());
                    for i in 0..list.len() {
                        builder.append(validity.slice(i * vl, vl).count_set_bits() < vl);
                    }
                    builder.finish()
                }
                None => BooleanBuffer::new_unset(list.len()),
            };
            Ok(mask_with_list_nulls(buf, list.nulls()))
        }
        dt => exec_err!("compute_row_has_nulls: unsupported data type {dt}"),
    }
}

/// Computes per-row null presence for `List` and `LargeList` arrays.
fn generic_list_row_has_nulls<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
) -> Result<BooleanBuffer> {
    let buf = match list.values().nulls() {
        Some(nulls) => {
            let validity = nulls.inner();
            let offsets = list.offsets();
            let mut builder = BooleanBufferBuilder::new(list.len());
            for i in 0..list.len() {
                let s = offsets[i].as_usize();
                let len = offsets[i + 1].as_usize() - s;
                builder.append(validity.slice(s, len).count_set_bits() < len);
            }
            builder.finish()
        }
        None => BooleanBuffer::new_unset(list.len()),
    };
    Ok(mask_with_list_nulls(buf, list.nulls()))
}

/// Rows where the list itself is null should not be marked as "has nulls".
fn mask_with_list_nulls(
    buf: BooleanBuffer,
    list_nulls: Option<&NullBuffer>,
) -> BooleanBuffer {
    match list_nulls {
        Some(n) => &buf & n.inner(),
        None => buf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Float32Array, Float64Array, Int32Array, Int32Builder, ListArray, ListBuilder,
    };
    use arrow::buffer::OffsetBuffer;
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

    /// One valid row of a zero-width list, without a null buffer to infer the length from.
    fn zero_width_list(field: &FieldRef) -> FixedSizeListArray {
        FixedSizeListArray::try_new_with_length(
            Arc::clone(field),
            0,
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            None,
            1,
        )
        .unwrap()
    }

    #[test]
    fn zero_width_fixed_size_list_keeps_its_row() {
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let haystack = zero_width_list(&field);
        assert_eq!(haystack.len(), 1);

        let result = SparkArrayContains::new()
            .invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Array(Arc::new(haystack)),
                    ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
                ],
                arg_fields: vec![],
                number_rows: 1,
                return_field: Arc::new(Field::new("result", DataType::Boolean, true)),
                config_options: Default::default(),
            })
            .unwrap()
            .to_array(1)
            .unwrap();
        assert_eq!(result.as_boolean(), &BooleanArray::from(vec![Some(false)]));
    }

    #[test]
    fn zero_width_fixed_size_list_nested_in_a_list_is_not_rebuilt() {
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let inner = Arc::new(Field::new(
            "inner",
            DataType::FixedSizeList(Arc::clone(&item), 0),
            true,
        ));
        let list = ListArray::try_new(
            inner,
            OffsetBuffer::new(vec![0, 1].into()),
            Arc::new(zero_width_list(&item)),
            None,
        )
        .unwrap();
        let array: ArrayRef = Arc::new(list);

        let normalized = normalize_array(&array).unwrap();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized.as_list::<i32>().values().len(), 1);
    }
}
