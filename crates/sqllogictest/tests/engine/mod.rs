// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A sqllogictest engine that prints floating-point results at full precision.
//!
//! DataFusion's own engine rounds a top-level DOUBLE to 15 decimal places
//! (12 for FLOAT), which hides exactly the last-digit differences that matter
//! when emulating Spark's `java.lang.StrictMath`. This engine prints the
//! shortest decimal text that round-trips to the same value instead, so
//! `0.6931471805599453` and `2.9999999999999996` appear as such. Every other
//! type is formatted by DataFusion's engine, unchanged.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, AsArray, StringArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{
    DataType, Field, Float32Type, Float64Type, Schema, SchemaRef,
};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::physical_plan::common::collect;
use datafusion::physical_plan::execute_stream;
use datafusion::prelude::SessionContext;
use datafusion_sqllogictest::{
    DFColumnType, DFOutput, DFSqlLogicTestError, convert_batches, convert_schema_to_types,
};
use sqllogictest::DBOutput;

pub struct SparkDataFusion {
    ctx: SessionContext,
}

impl SparkDataFusion {
    pub fn new(ctx: SessionContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl sqllogictest::AsyncDB for SparkDataFusion {
    type Error = DFSqlLogicTestError;
    type ColumnType = DFColumnType;

    async fn run(&mut self, sql: &str) -> Result<DFOutput, DFSqlLogicTestError> {
        let df = self.ctx.sql(sql).await?;
        let task_ctx = Arc::new(df.task_ctx());
        let plan = df.create_physical_plan().await?;
        let stream = execute_stream(plan, task_ctx)?;
        // The column types come from the real schema, so a DOUBLE column is
        // still checked against `R` after its values are rendered as text.
        let types = convert_schema_to_types(stream.schema().fields());
        let schema = stream.schema();
        let batches = collect(stream).await?;

        let text_schema = floats_as_text_schema(&schema);
        let batches = batches
            .into_iter()
            .map(|batch| floats_as_text(&text_schema, &batch))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = convert_batches(&text_schema, batches, true)?;

        if rows.is_empty() && types.is_empty() {
            Ok(DBOutput::StatementComplete(0))
        } else {
            Ok(DBOutput::Rows { types, rows })
        }
    }

    fn engine_name(&self) -> &str {
        "DataFusion"
    }

    async fn sleep(dur: Duration) {
        tokio::time::sleep(dur).await;
    }

    async fn shutdown(&mut self) {}
}

/// The float type a column holds at the top level, looking through a dictionary.
fn float_type(data_type: &DataType) -> Option<&DataType> {
    match data_type {
        DataType::Float32 | DataType::Float64 => Some(data_type),
        DataType::Dictionary(_, value) => float_type(value),
        _ => None,
    }
}

fn floats_as_text_schema(schema: &SchemaRef) -> SchemaRef {
    let fields = schema
        .fields()
        .iter()
        .map(|field| match float_type(field.data_type()) {
            Some(_) => Arc::new(Field::new(field.name(), DataType::Utf8, true)),
            None => Arc::clone(field),
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()))
}

fn floats_as_text(
    text_schema: &SchemaRef,
    batch: &RecordBatch,
) -> Result<RecordBatch, DFSqlLogicTestError> {
    let columns = batch
        .columns()
        .iter()
        .map(|column| match float_type(column.data_type()) {
            Some(float) => {
                let column = cast(column, float)?;
                Ok(match float {
                    DataType::Float32 => {
                        render(column.as_primitive::<Float32Type>().iter(), format_f32)
                    }
                    _ => render(column.as_primitive::<Float64Type>().iter(), format_f64),
                })
            }
            None => Ok(Arc::clone(column)),
        })
        .collect::<Result<Vec<_>, DFSqlLogicTestError>>()?;
    Ok(RecordBatch::try_new(Arc::clone(text_schema), columns)?)
}

fn render<T>(
    values: impl Iterator<Item = Option<T>>,
    format: fn(T) -> String,
) -> ArrayRef {
    Arc::new(values.map(|v| v.map(format)).collect::<StringArray>())
}

/// The shortest decimal text that round-trips to `value`, without an exponent.
/// Like DataFusion's engine, NaN ignores its sign and a whole number has no
/// trailing `.0`. Unlike it, `-0.0` keeps its sign, since Spark distinguishes it.
fn format_f64(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        // Rust's `Display` for floats is the shortest round-trip text and
        // never uses an exponent.
        value.to_string()
    }
}

fn format_f32(value: f32) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        value.to_string()
    }
}
