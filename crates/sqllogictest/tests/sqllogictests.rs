// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Runs the `.slt` files under `test_files` against a DataFusion session with the Spark
//! functions registered and the Spark SQL dialect enabled.
//!
//! Usage: `cargo test -p datafusion-spark-sqllogictest -- [FILTER]`, where `FILTER` is an
//! optional substring of the test file path. Pass `--complete` to rewrite the expected results
//! in the matching files with the actual results instead of checking them.

mod engine;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use datafusion::common::config::Dialect;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_spark::SessionStateBuilderSpark;
use datafusion_sqllogictest::{df_value_validator, value_normalizer};
use engine::SparkDataFusion;
use sqllogictest::strict_column_validator;

fn collect_slt_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_slt_files(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "slt") {
            files.push(path);
        }
    }
    Ok(())
}

fn new_context() -> SessionContext {
    let mut config = SessionConfig::new().with_target_partitions(4);
    config.options_mut().sql_parser.dialect = Dialect::Spark;
    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_spark_features()
        .build();
    SessionContext::new_with_state(state)
}

async fn run_file(path: PathBuf, complete: bool) -> Result<(), String> {
    let mut runner =
        sqllogictest::Runner::new(|| async { Ok(SparkDataFusion::new(new_context())) });
    if complete {
        return runner
            .update_test_file(
                &path,
                " ",
                df_value_validator,
                value_normalizer,
                strict_column_validator,
            )
            .await
            .map_err(|e| e.to_string());
    }
    runner.with_column_validator(strict_column_validator);
    runner.with_normalizer(value_normalizer);
    runner.with_validator(df_value_validator);
    runner
        .run_file_async(&path)
        .await
        .map_err(|e| e.to_string())
}

#[tokio::main]
async fn main() -> ExitCode {
    let complete = std::env::args().any(|arg| arg == "--complete");
    // Ignore other flags such as `--nocapture` that `cargo test` may pass through.
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .collect();

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_files");
    let mut files = vec![];
    collect_slt_files(&root, &mut files).expect("failed to list test files");
    files.sort();

    let mut handles = vec![];
    for path in files {
        let relative = path.strip_prefix(&root).unwrap().to_path_buf();
        if !filters
            .iter()
            .all(|f| relative.to_string_lossy().contains(f.as_str()))
        {
            continue;
        }
        handles.push((relative.clone(), tokio::spawn(run_file(path, complete))));
    }

    let total = handles.len();
    let mut failed = 0;
    for (relative, handle) in handles {
        match handle.await.expect("test task panicked") {
            Ok(()) => println!("PASS {}", relative.display()),
            Err(e) => {
                failed += 1;
                println!("FAIL {}\n{e}\n", relative.display());
            }
        }
    }
    println!("{} of {total} test files passed", total - failed);
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
