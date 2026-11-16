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
//! skips every `onlyif` record that is not for `spark`. Whole files that cannot run in
//! `evaluate` mode yet are listed in [`EVALUATE_KNOWN_FAILURES`].
//!
//! Usage: `cargo test -p datafusion-spark-sqllogictest -- [FILTER]... [--mode=MODE]`, where
//! each `FILTER` is a substring of the test file path and `MODE` is `simplify` or
//! `evaluate` (both by default).

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
use datafusion_sqllogictest::{DataFusion, df_value_validator, value_normalizer};
use indicatif::ProgressBar;
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

/// Files that fail in [`Mode::Evaluate`] because a function has no working
/// `invoke_with_args` yet, with the issue that tracks it. A failure in one of these
/// files is reported but does not fail the run. A pass fails the run, so the entry is
/// removed once the issue is fixed.
const EVALUATE_KNOWN_FAILURES: [(&str, &str); 7] = [
    ("conditional/if.slt", "#305"),
    ("datetime/date_diff.slt", "#307"),
    ("datetime/date_part.slt", "#308"),
    ("datetime/date_trunc.slt", "#309"),
    ("datetime/trunc.slt", "#311"),
    ("datetime/add_months.slt", "#312"),
    ("string/base64.slt", "#313"),
];

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

async fn run_file(
    path: PathBuf,
    relative_path: PathBuf,
    mode: Mode,
) -> Result<(), String> {
    // The DataFusion engine formats floats the way Spark does for paths under `spark/`
    let engine_path = Path::new("spark").join(&relative_path);
    let mut runner = sqllogictest::Runner::new(|| async {
        Ok(DataFusion::new(
            new_context(mode),
            engine_path.clone(),
            ProgressBar::hidden(),
        ))
    });
    runner.add_label(mode.label());
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
    let modes: Vec<Mode> = match args.iter().find_map(|arg| arg.strip_prefix("--mode=")) {
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
            let task = tokio::spawn(run_file(path.clone(), relative.clone(), mode));
            handles.push((relative.clone(), mode, task));
        }
    }

    let total = handles.len();
    let mut failed = 0;
    for (relative, mode, handle) in handles {
        let name = format!("[{}] {}", mode.label(), relative.display());
        let known_failure = EVALUATE_KNOWN_FAILURES
            .iter()
            .find(|(file, _)| mode == Mode::Evaluate && Path::new(file) == relative)
            .map(|(_, issue)| *issue);
        // A panic in one file (for example a function indexing past the end of an
        // array) is reported as that file failing, so the other results still print.
        match (
            handle.await.map_err(|e| e.to_string()).and_then(|r| r),
            known_failure,
        ) {
            (Ok(()), None) => println!("PASS {name}"),
            (Err(_), Some(issue)) => {
                println!("XFAIL {name} (known failure, see {issue})")
            }
            (Ok(()), Some(issue)) => {
                failed += 1;
                println!(
                    "FAIL {name}\nexpected to fail ({issue}) but passed, \
                     remove it from EVALUATE_KNOWN_FAILURES\n"
                );
            }
            (Err(e), None) => {
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
