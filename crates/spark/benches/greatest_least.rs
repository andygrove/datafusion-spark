// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::PrimitiveArray;
use arrow::datatypes::{Field, Float64Type, Int64Type};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
use datafusion_spark::function::math::{greatest, least};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::sync::Arc;

const ROWS: usize = 8192;
const NULL_DENSITY: f32 = 0.1;

fn int64_column(rng: &mut StdRng) -> ColumnarValue {
    let values: PrimitiveArray<Int64Type> = (0..ROWS)
        .map(|_| (rng.random::<f32>() >= NULL_DENSITY).then(|| rng.random()))
        .collect();
    ColumnarValue::Array(Arc::new(values))
}

/// Includes NaN and -0.0, which Spark orders differently from IEEE 754.
fn float64_column(rng: &mut StdRng) -> ColumnarValue {
    let values: PrimitiveArray<Float64Type> = (0..ROWS)
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
    ColumnarValue::Array(Arc::new(values))
}

fn bench(c: &mut Criterion, udf: &ScalarUDF, label: &str, args: &[ColumnarValue]) {
    let arg_fields: Vec<_> = args
        .iter()
        .map(|arg| Arc::new(Field::new("a", arg.data_type(), true)))
        .collect();
    let scalar_arguments = vec![None; args.len()];
    let return_field = udf
        .return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap();
    let config = Arc::new(ConfigOptions::default());
    c.bench_function(&format!("{} {label}", udf.name()), |b| {
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
    let ints = [int64_column(rng), int64_column(rng)];
    let floats = [
        float64_column(rng),
        float64_column(rng),
        float64_column(rng),
    ];
    let greatest = greatest();
    bench(c, &greatest, "2 x int64", &ints);
    bench(c, &greatest, "2 x float64", &floats[..2]);
    bench(c, &greatest, "3 x float64", &floats);
    bench(c, &least(), "3 x float64", &floats);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
