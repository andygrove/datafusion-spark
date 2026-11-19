// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{ArrayRef, ListArray, PrimitiveArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{Field, Float64Type, Int64Type};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::ScalarValue;
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDF};
use datafusion_spark::function::array::array_remove;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;

const ROWS: usize = 8192;
const MAX_LIST_LEN: usize = 32;
const NULL_DENSITY: f32 = 0.1;

fn list_of(values: ArrayRef, rng: &mut StdRng) -> ColumnarValue {
    let lengths: Vec<usize> = (0..ROWS)
        .map(|_| rng.random_range(0..MAX_LIST_LEN))
        .collect();
    let values = values.slice(0, lengths.iter().sum());
    let field = Arc::new(Field::new_list_field(values.data_type().clone(), true));
    ColumnarValue::Array(Arc::new(ListArray::new(
        field,
        OffsetBuffer::from_lengths(lengths),
        values,
        None,
    )))
}

/// Small values, so that the element matches often. Includes NaN and -0.0.
fn float64_values(rng: &mut StdRng) -> ArrayRef {
    let values: PrimitiveArray<Float64Type> = (0..ROWS * MAX_LIST_LEN)
        .map(|_| {
            (rng.random::<f32>() >= NULL_DENSITY).then(|| match rng.random_range(0..20) {
                0 => f64::NAN,
                1 => -0.0,
                n => f64::from(n % 8),
            })
        })
        .collect();
    Arc::new(values)
}

fn int64_values(rng: &mut StdRng) -> ArrayRef {
    let values: PrimitiveArray<Int64Type> = (0..ROWS * MAX_LIST_LEN)
        .map(|_| (rng.random::<f32>() >= NULL_DENSITY).then(|| rng.random_range(0..8)))
        .collect();
    Arc::new(values)
}

fn bench(c: &mut Criterion, udf: &ScalarUDF, label: &str, args: [ColumnarValue; 2]) {
    let arg_fields: Vec<_> = args
        .iter()
        .map(|arg| Arc::new(Field::new("a", arg.data_type(), true)))
        .collect();
    let return_field = Arc::new(Field::new("r", args[0].data_type(), true));
    let config = Arc::new(ConfigOptions::default());
    c.bench_function(&format!("array_remove {label}"), |b| {
        b.iter(|| {
            black_box(
                udf.invoke_with_args(ScalarFunctionArgs {
                    args: args.to_vec(),
                    arg_fields: arg_fields.clone(),
                    number_rows: ROWS,
                    return_field: Arc::clone(&return_field),
                    config_options: Arc::clone(&config),
                })
                .unwrap(),
            )
        })
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let rng = &mut StdRng::seed_from_u64(42);
    let udf = array_remove();
    let floats = list_of(float64_values(rng), rng);
    let ints = list_of(int64_values(rng), rng);
    let float_column =
        ColumnarValue::Array(Arc::new(PrimitiveArray::<Float64Type>::from(
            (0..ROWS).map(|i| Some((i % 8) as f64)).collect::<Vec<_>>(),
        )));
    let scalar = |v: ScalarValue| ColumnarValue::Scalar(v);
    bench(
        c,
        &udf,
        "float64 constant",
        [floats.clone(), scalar(ScalarValue::Float64(Some(3.0)))],
    );
    bench(
        c,
        &udf,
        "float64 constant NaN",
        [floats.clone(), scalar(ScalarValue::Float64(Some(f64::NAN)))],
    );
    bench(c, &udf, "float64 column", [floats, float_column]);
    bench(
        c,
        &udf,
        "int64 constant",
        [ints, scalar(ScalarValue::Int64(Some(3)))],
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
