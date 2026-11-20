// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::float_normalize::normalize_floats;
use crate::function::nullability::{SparkNullability, spark_nullability};
use arrow::array::{
    Array, AsArray, BooleanArray, BooleanBufferBuilder, GenericListArray, OffsetSizeTrait,
};
use arrow::buffer::{BooleanBuffer, NullBuffer};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, exec_err};
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions_nested::array_has::array_has_udf;
use std::sync::Arc;

/// Spark's `ArrayContains` is a null-intolerant `BinaryExpression`. It is also nullable
/// when the array can hold NULL elements.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

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

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let contains_null = match args.arg_fields[0].data_type() {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::FixedSizeList(field, _) => field.is_nullable(),
            _ => true,
        };
        let nullable = NULLABILITY.nullable(args.arg_fields) || contains_null;
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Boolean,
            nullable,
        )))
    }

    spark_nullability!(NULLABILITY);

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
    use arrow::array::{FixedSizeListArray, Float64Array};
    use datafusion_common::ScalarValue;

    fn return_field_nullable(array_nullable: bool, element_nullable: bool) -> bool {
        let element = Arc::new(Field::new("item", DataType::Int64, element_nullable));
        let arg_fields = [
            Arc::new(Field::new("a", DataType::List(element), array_nullable)),
            Arc::new(Field::new("v", DataType::Int64, false)),
        ];
        SparkArrayContains::new()
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &arg_fields,
                scalar_arguments: &[None, None],
            })
            .unwrap()
            .is_nullable()
    }

    #[test]
    fn return_field_is_nullable_when_the_array_can_hold_nulls() {
        assert!(!return_field_nullable(false, false));
        assert!(return_field_nullable(true, false));
        assert!(return_field_nullable(false, true));
    }

    #[test]
    fn zero_width_fixed_size_list_keeps_its_row() {
        // One valid row of a zero-width list, without a null buffer to infer the length from.
        let field = Arc::new(Field::new("item", DataType::Float64, true));
        let haystack = FixedSizeListArray::try_new_with_length(
            Arc::clone(&field),
            0,
            Arc::new(Float64Array::from(Vec::<f64>::new())),
            None,
            1,
        )
        .unwrap();
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
}
