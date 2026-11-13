// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use super::array_contains::normalize_array;
use arrow::array::{
    Array, ArrayRef, GenericListArray, Int64Array, OffsetSizeTrait, make_comparator,
};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion_common::{Result, ScalarValue, exec_err};
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
        array, as a `BIGINT`. Unlike the DataFusion built-in, which returns NULL when the element \
        is not found, this returns 0. The result is NULL only when the array or the element is \
        NULL. NULL elements inside the array never match. For floating point values, NaN is \
        equal to NaN and -0.0 is equal to 0.0.",
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
            signature: Signature::array_and_element(Volatility::Immutable),
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

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let is_scalar = args
            .args
            .iter()
            .all(|a| matches!(a, ColumnarValue::Scalar(_)));
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let [array, element] = arrays.as_slice() else {
            return exec_err!("array_position expects exactly two arguments");
        };

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
    let list = array
        .as_any()
        .downcast_ref::<GenericListArray<O>>()
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Internal("expected list array".into())
        })?;
    // Canonicalize NaN and -0.0 at any depth so that the comparator follows Spark's float
    // equality. The comparator treats NULLs nested inside the compared values as equal.
    let values = normalize_array(list.values())?;
    let element = normalize_array(element)?;
    let nulls = NullBuffer::union(list.nulls(), element.nulls());
    let cmp = make_comparator(values.as_ref(), element.as_ref(), SortOptions::default())?;

    let result: Vec<i64> = list
        .offsets()
        .windows(2)
        .enumerate()
        .map(|(row, w)| {
            if nulls.as_ref().is_some_and(|n| n.is_null(row)) {
                return 0;
            }
            let (start, end) = (w[0].as_usize(), w[1].as_usize());
            (start..end)
                .find(|&i| values.is_valid(i) && cmp(i, row).is_eq())
                .map_or(0, |i| (i - start + 1) as i64)
        })
        .collect();

    Ok(Arc::new(Int64Array::new(ScalarBuffer::from(result), nulls)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{FixedSizeListArray, Float64Array, Int32Array, ListArray};
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Field, Int32Type};

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
