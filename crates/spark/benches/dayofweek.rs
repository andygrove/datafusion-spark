// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Measure `dayofweek` and `weekday` over Date32 and Date64 columns at several
//! null densities.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{ArrayRef, Date32Array, Date64Array};
use arrow::datatypes::{DataType, Field};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};
use datafusion_spark::function::datetime::dayofweek::SparkDayOfWeek;
use datafusion_spark::function::datetime::weekday::SparkWeekDay;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const ROWS: usize = 8192;

fn days(null_density: f32) -> Vec<Option<i32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..ROWS)
        .map(|_| {
            (rng.random::<f32>() >= null_density)
                .then(|| rng.random_range(-1_000_000..1_000_000))
        })
        .collect()
}

fn inputs(null_density: f32) -> [(&'static str, ArrayRef); 2] {
    let days = days(null_density);
    let date64 = days.iter().map(|d| d.map(|d| i64::from(d) * 86_400_000));
    [
        ("date32", Arc::new(Date32Array::from(days.clone()))),
        ("date64", Arc::new(Date64Array::from_iter(date64))),
    ]
}

fn bench_udf(c: &mut Criterion, udf: &dyn ScalarUDFImpl) {
    let config_options = Arc::new(ConfigOptions::default());
    let mut group = c.benchmark_group(udf.name());
    group.throughput(Throughput::Elements(ROWS as u64));
    for (null_density, nulls) in [
        (0.0, "no_nulls"),
        (0.1, "10pct_nulls"),
        (0.5, "50pct_nulls"),
    ] {
        for (type_name, array) in inputs(null_density) {
            let args = vec![ColumnarValue::Array(array)];
            let arg_fields = vec![Arc::new(Field::new("d", args[0].data_type(), true))];
            group.bench_function(BenchmarkId::new(type_name, nulls), |b| {
                b.iter(|| {
                    black_box(
                        udf.invoke_with_args(ScalarFunctionArgs {
                            args: args.clone(),
                            arg_fields: arg_fields.clone(),
                            number_rows: ROWS,
                            return_field: Arc::new(Field::new(
                                "f",
                                DataType::Int32,
                                true,
                            )),
                            config_options: Arc::clone(&config_options),
                        })
                        .unwrap(),
                    )
                })
            });
        }
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_udf(c, &SparkWeekDay::new());
    bench_udf(c, &SparkDayOfWeek::new());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
