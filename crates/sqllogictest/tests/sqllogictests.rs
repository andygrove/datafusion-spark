// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Runs the `.slt` files under `test_files` against a DataFusion session with the Spark
//! functions registered and the Spark SQL dialect enabled.
//!
//! Each file runs twice, once in each [`Mode`]:
//!
//! - `simplify`: the default optimizer rules. Expressions whose arguments are all
//!   literals are constant-folded during planning, which calls a function's `simplify`
//!   (if it has one) and then evaluates it against a one-row batch.
//! - `evaluate`: the same rules without `simplify_expressions`, so every expression,
//!   including `SELECT f(1)`, is planned as written and run through `invoke_with_args`
//!   during execution. This is the path engines such as Comet use, since they build
//!   physical expressions directly and never call `simplify`. DataFusion's own
//!   functions that can only run as a rewrite, such as `arrow_cast`, are still
//!   rewritten (see [`RewriteCoreFunctions`]).
//!
//! A record that cannot run in `evaluate` mode yet uses `skipif evaluate`, with a comment
//! that links the issue. Prefer it over `onlyif simplify`, because the PySpark runner
//! skips every `onlyif` record that is not for `spark`.
//!
//! Usage: `cargo test -p datafusion-spark-sqllogictest -- [FILTER]... [--mode=MODE]
//! [--complete]`, where each `FILTER` is a substring of the test file path and `MODE` is
//! `simplify` or `evaluate` (both by default).
//!
//! `--complete` rewrites the expected output of each matching file with the results of
//! running it, in `simplify` mode unless `--mode` says otherwise. Review the diff before
//! committing it, since it records whatever the functions return.

mod engine;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use datafusion::common::config::Dialect;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchema, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion::logical_expr::utils::merge_schema;
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::simplify_expressions::SimplifyExpressions;
use datafusion::optimizer::{Optimizer, OptimizerConfig, OptimizerRule};
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

/// How expressions in a test file reach the function implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Default optimizer rules, so literal-only expressions are constant-folded.
    Simplify,
    /// No `simplify_expressions` rule, so every expression runs through `invoke_with_args`.
    Evaluate,
}

impl Mode {
    const ALL: [Mode; 2] = [Mode::Simplify, Mode::Evaluate];

    /// The `.slt` label for `onlyif` and `skipif`.
    fn label(self) -> &'static str {
        match self {
            Mode::Simplify => "simplify",
            Mode::Evaluate => "evaluate",
        }
    }
}

/// DataFusion functions whose `invoke_with_args` always errors because they are meant
/// to be rewritten by `simplify` (for example `arrow_cast` becomes `CAST`). The tests
/// use them to build typed inputs, so [`Mode::Evaluate`] still rewrites them.
const SIMPLIFY_ONLY_CORE_FUNCTIONS: [&str; 5] = [
    "arrow_cast",
    "arrow_try_cast",
    "cast_to_type",
    "try_cast_to_type",
    "nvl2",
];

/// Applies `simplify` to [`SIMPLIFY_ONLY_CORE_FUNCTIONS`] only, leaving every other
/// expression, including literal-only calls of the Spark functions, as written.
#[derive(Debug)]
struct RewriteCoreFunctions;

impl OptimizerRule for RewriteCoreFunctions {
    fn name(&self) -> &str {
        "rewrite_core_functions"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let schema = match &plan {
            LogicalPlan::TableScan(scan) => DFSchema::try_from_qualified_schema(
                scan.table_name.clone(),
                &scan.source.schema(),
            )?,
            _ => merge_schema(&plan.inputs()),
        };
        let info = SimplifyContext::builder()
            .with_schema(Arc::new(schema))
            .with_config_options(config.options())
            .with_query_execution_start_time(config.query_execution_start_time())
            .build();
        let name_preserver = NamePreserver::new(&plan);
        plan.map_expressions(|expr| {
            let name = name_preserver.save(&expr);
            let expr = expr.transform_up(|expr| match expr {
                Expr::ScalarFunction(f)
                    if SIMPLIFY_ONLY_CORE_FUNCTIONS.contains(&f.func.name()) =>
                {
                    match f.func.simplify(f.args, &info)? {
                        ExprSimplifyResult::Simplified(expr) => {
                            Ok(Transformed::yes(expr))
                        }
                        ExprSimplifyResult::Original(args) => Ok(Transformed::no(
                            Expr::ScalarFunction(ScalarFunction::new_udf(f.func, args)),
                        )),
                    }
                }
                _ => Ok(Transformed::no(expr)),
            })?;
            Ok(expr.update_data(|expr| name.restore(expr)))
        })
    }
}

fn new_context(mode: Mode) -> SessionContext {
    let mut config = SessionConfig::new().with_target_partitions(4);
    config.options_mut().sql_parser.dialect = Dialect::Spark;
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_spark_features();
    if mode == Mode::Evaluate {
        let simplify = SimplifyExpressions::new().name().to_string();
        let rules = Optimizer::new()
            .rules
            .into_iter()
            .map(|rule| {
                if rule.name() == simplify {
                    Arc::new(RewriteCoreFunctions) as _
                } else {
                    rule
                }
            })
            .collect();
        builder = builder.with_optimizer_rules(rules);
    }
    SessionContext::new_with_state(builder.build())
}

/// Runs the file at `path`, or with `complete`, rewrites its expected output with the
/// results of this run.
async fn run_file(path: PathBuf, mode: Mode, complete: bool) -> Result<(), String> {
    let mut runner = sqllogictest::Runner::new(|| async {
        Ok(SparkDataFusion::new(new_context(mode)))
    });
    runner.add_label(mode.label());
    if complete {
        // Records skipped for this mode's label are written back unchanged.
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let complete = args.iter().any(|arg| arg == "--complete");
    let modes: Vec<Mode> = match args.iter().find_map(|arg| arg.strip_prefix("--mode=")) {
        // Completion writes one set of results, so it runs one mode. Both modes must
        // produce the same results, which a normal run then checks.
        None if complete => vec![Mode::Simplify],
        None => Mode::ALL.to_vec(),
        Some(name) => match Mode::ALL.into_iter().find(|m| m.label() == name) {
            Some(mode) => vec![mode],
            None => {
                eprintln!("unknown mode {name:?}, expected `simplify` or `evaluate`");
                return ExitCode::FAILURE;
            }
        },
    };
    // Ignore flags such as `--nocapture` that `cargo test` may pass through.
    let filters: Vec<String> = args
        .into_iter()
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
        for &mode in &modes {
            let task = tokio::spawn(run_file(path.clone(), mode, complete));
            handles.push((relative.clone(), mode, task));
        }
    }

    let total = handles.len();
    let mut failed = 0;
    for (relative, mode, handle) in handles {
        let name = format!("[{}] {}", mode.label(), relative.display());
        // A panic in one file (for example a function indexing past the end of an
        // array) is reported as that file failing, so the other results still print.
        match handle.await.map_err(|e| e.to_string()).and_then(|r| r) {
            Ok(()) if complete => println!("COMPLETED {name}"),
            Ok(()) => println!("PASS {name}"),
            Err(e) => {
                failed += 1;
                println!("FAIL {name}\n{e}\n");
            }
        }
    }
    println!("{} of {total} test runs passed", total - failed);
    if failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
