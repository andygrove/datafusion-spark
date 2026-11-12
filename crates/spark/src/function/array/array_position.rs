// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{
    Array, ArrayRef, AsArray, GenericListArray, Int64Array, OffsetSizeTrait,
    make_comparator,
};
use arrow::buffer::{NullBuffer, ScalarBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Float32Type, Float64Type};
use datafusion_common::{Result, ScalarValue, exec_err, not_impl_err};
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

type Equality = Box<dyn Fn(usize, usize) -> bool + Send + Sync>;

fn has_float(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float32 | DataType::Float64 => true,
        DataType::List(f) | DataType::LargeList(f) | DataType::FixedSizeList(f, _) => {
            has_float(f.data_type())
        }
        DataType::Struct(fields) => fields.iter().any(|f| has_float(f.data_type())),
        DataType::Map(f, _) => has_float(f.data_type()),
        _ => false,
    }
}

/// Wraps a non-null equality so that NULL equals NULL and NULL differs from any value, which
/// is how Spark compares values nested inside arrays and structs.
fn null_aware(
    left: &ArrayRef,
    right: &ArrayRef,
    eq: impl Fn(usize, usize) -> bool + Send + Sync + 'static,
) -> Equality {
    let (left, right) = (Arc::clone(left), Arc::clone(right));
    Box::new(move |i, j| match (left.is_null(i), right.is_null(j)) {
        (true, true) => true,
        (false, false) => eq(i, j),
        _ => false,
    })
}

/// Builds `eq(left_index, right_index)` in Spark's equality, where NaN equals NaN and -0.0
/// equals 0.0 at any depth. Subtrees without a float use Arrow's comparator.
fn equality(left: &ArrayRef, right: &ArrayRef) -> Result<Equality> {
    macro_rules! float_eq {
        ($t:ty) => {{
            let (l, r) = (
                left.as_primitive::<$t>().clone(),
                right.as_primitive::<$t>().clone(),
            );
            null_aware(left, right, move |i, j| {
                let (a, b) = (l.value(i), r.value(j));
                a == b || (a.is_nan() && b.is_nan())
            })
        }};
    }
    Ok(match left.data_type() {
        DataType::Float32 => float_eq!(Float32Type),
        DataType::Float64 => float_eq!(Float64Type),
        DataType::List(_) => list_equality::<i32>(left, right)?,
        DataType::LargeList(_) => list_equality::<i64>(left, right)?,
        DataType::Struct(_) => {
            let (l, r) = (left.as_struct(), right.as_struct());
            let fields = l
                .columns()
                .iter()
                .zip(r.columns())
                .map(|(lc, rc)| equality(lc, rc))
                .collect::<Result<Vec<_>>>()?;
            null_aware(left, right, move |i, j| fields.iter().all(|eq| eq(i, j)))
        }
        dt if has_float(dt) => {
            return not_impl_err!("array_position does not support {dt} with floats");
        }
        _ => {
            let cmp =
                make_comparator(left.as_ref(), right.as_ref(), SortOptions::default())?;
            Box::new(move |i, j| cmp(i, j).is_eq())
        }
    })
}

fn list_equality<O: OffsetSizeTrait>(
    left: &ArrayRef,
    right: &ArrayRef,
) -> Result<Equality> {
    let (l, r) = (left.as_list::<O>().clone(), right.as_list::<O>().clone());
    let inner = equality(l.values(), r.values())?;
    Ok(null_aware(left, right, move |i, j| {
        let (lo, ro) = (l.value_offsets(), r.value_offsets());
        let (ls, le) = (lo[i].as_usize(), lo[i + 1].as_usize());
        let (rs, re) = (ro[j].as_usize(), ro[j + 1].as_usize());
        le - ls == re - rs && (0..le - ls).all(|k| inner(ls + k, rs + k))
    }))
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
    let values = list.values();
    let nulls = NullBuffer::union(list.nulls(), element.nulls());
    let eq = equality(values, element)?;

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
                .find(|&i| values.is_valid(i) && eq(i, row))
                .map_or(0, |i| (i - start + 1) as i64)
        })
        .collect();

    Ok(Arc::new(Int64Array::new(ScalarBuffer::from(result), nulls)))
}
