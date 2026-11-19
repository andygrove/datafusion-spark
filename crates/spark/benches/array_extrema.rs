// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{ArrayRef, ListArray, PrimitiveArray, StringArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{Field, Float64Type, Int64Type};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
use datafusion_spark::function::array::{array_max, array_min};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;

const ROWS: usize = 8192;
const MAX_LIST_LEN: usize = 32;
const NULL_DENSITY: f32 = 0.1;

fn list_of(values: ArrayRef, lengths: Vec<usize>) -> ColumnarValue {
    let field = Arc::new(Field::new_list_field(values.data_type().clone(), true));
    ColumnarValue::Array(Arc::new(ListArray::new(
        field,
        OffsetBuffer::from_lengths(lengths),
        values,
        None,
    )))
}

fn random_lengths(rng: &mut StdRng) -> Vec<usize> {
    (0..ROWS)
        .map(|_| rng.random_range(0..MAX_LIST_LEN))
        .collect()
}

fn int64_lists(rng: &mut StdRng) -> ColumnarValue {
    let lengths = random_lengths(rng);
    let values: PrimitiveArray<Int64Type> = (0..lengths.iter().sum())
        .map(|_| (rng.random::<f32>() >= NULL_DENSITY).then(|| rng.random()))
        .collect();
    list_of(Arc::new(values), lengths)
}

fn float64_lists(rng: &mut StdRng) -> ColumnarValue {
    let lengths = random_lengths(rng);
    let values: PrimitiveArray<Float64Type> = (0..lengths.iter().sum())
        .map(|_| {
            (rng.random::<f32>() >= NULL_DENSITY).then(|| {
                match rng.random_range(0..100) {
                    0 => f64::NAN,
                    1 => -0.0,
                    _ => rng.random::<f64>() * 2.0 - 1.0,
                }
            })
        })
        .collect();
    list_of(Arc::new(values), lengths)
}

fn string_lists(rng: &mut StdRng) -> ColumnarValue {
    let lengths = random_lengths(rng);
    let values: StringArray = (0..lengths.iter().sum())
        .map(|_| {
            (rng.random::<f32>() >= NULL_DENSITY)
                .then(|| format!("value-{}", rng.random::<u32>()))
        })
        .collect();
    list_of(Arc::new(values), lengths)
}

fn bench(c: &mut Criterion, udf: &ScalarUDF, label: &str, input: &ColumnarValue) {
    let arg_fields = vec![Arc::new(Field::new("a", input.data_type(), true))];
    let return_field = udf
        .return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &[None],
        })
        .unwrap();
    let config = Arc::new(ConfigOptions::default());
    c.bench_function(&format!("{} {label}", udf.name()), |b| {
        b.iter(|| {
            black_box(
                udf.invoke_with_args(ScalarFunctionArgs {
                    args: vec![input.clone()],
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
    let inputs = [
        ("int64", int64_lists(rng)),
        ("float64", float64_lists(rng)),
        ("utf8", string_lists(rng)),
    ];
    let max = array_max();
    let min = array_min();
    for (label, input) in &inputs {
        bench(c, &max, label, input);
    }
    bench(c, &min, "float64", &inputs[1].1);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
