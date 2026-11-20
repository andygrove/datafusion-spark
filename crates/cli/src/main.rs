// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `datafusion-spark-cli`: the DataFusion CLI with the Spark-compatible functions
//! registered and the Spark SQL dialect enabled.
//!
//! The shell itself (the REPL, running files and commands, output formats and
//! object stores) comes from the `datafusion-cli` library.

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use clap::{ArgAction, Parser};
use datafusion::common::config::{ConfigNonZeroUsize, Dialect};
use datafusion::common::{DataFusionError, Result};
use datafusion::config::ConfigOptions;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::logical_expr::ExplainFormat;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_cli::catalog::DynamicObjectStoreCatalog;
use datafusion_cli::exec;
use datafusion_cli::object_storage::instrumented::{
    InstrumentedObjectStoreMode, InstrumentedObjectStoreRegistry,
};
use datafusion_cli::object_storage::{StdinCarriesCommands, is_stdin_location};
use datafusion_cli::print_format::PrintFormat;
use datafusion_cli::print_options::{MaxRows, PrintOptions};
use datafusion_spark::SessionStateBuilderSpark;

#[derive(Debug, Parser)]
#[clap(version, about, long_about = None)]
struct Args {
    #[clap(
        short = 'c',
        long,
        num_args = 0..,
        help = "Execute the given command string(s), then exit",
        value_parser(parse_command)
    )]
    command: Vec<String>,

    #[clap(
        short,
        long,
        num_args = 0..,
        help = "Execute commands from file(s), then exit",
        value_parser(parse_valid_file)
    )]
    file: Vec<String>,

    #[clap(
        short = 'r',
        long,
        num_args = 0..,
        help = "Run the provided files on startup, before the REPL",
        value_parser(parse_valid_file),
        conflicts_with = "file"
    )]
    rc: Vec<String>,

    #[clap(
        short = 'p',
        long,
        help = "Path to your data, default to current directory",
        value_parser(parse_valid_data_dir)
    )]
    data_path: Option<String>,

    #[clap(
        short = 'b',
        long,
        help = "The batch size of each query, or use DataFusion default",
        value_parser(parse_batch_size)
    )]
    batch_size: Option<usize>,

    #[clap(
        long,
        default_value_t = true,
        action = ArgAction::Set,
        help = "Evaluate with ANSI mode on, like Spark 4's default \
                `spark.sql.ansi.enabled = true`. Pass `--ansi false` for Spark's legacy behavior"
    )]
    ansi: bool,

    #[clap(long, value_enum, default_value_t = PrintFormat::Automatic)]
    format: PrintFormat,

    #[clap(
        short,
        long,
        help = "Reduce printing other than the results and work quietly"
    )]
    quiet: bool,

    #[clap(
        long,
        help = "The max number of rows to display for 'Table' format\n[possible values: numbers(0/10/...), inf(no limit)]",
        default_value = "40"
    )]
    maxrows: MaxRows,

    #[clap(long, help = "Enables console syntax highlighting")]
    color: bool,
}

impl Args {
    /// Without `-c` or `-f` the CLI enters the REPL, which reads SQL from stdin.
    fn repl_mode(&self) -> bool {
        self.command.is_empty() && self.file.is_empty()
    }

    /// Whether stdin carries the session's SQL, so it cannot also back a
    /// `LOCATION '/dev/stdin'` table.
    fn reads_sql_from_stdin(&self) -> bool {
        self.repl_mode() || self.file.iter().any(|f| is_stdin_location(f))
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    if let Err(e) = main_inner().await {
        println!("Error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

async fn main_inner() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    if !args.quiet {
        println!(
            "DataFusion Spark CLI v{} (ANSI mode {})",
            env!("CARGO_PKG_VERSION"),
            if args.ansi { "on" } else { "off" }
        );
    }

    // Open every script before running any of them, so a file that cannot be opened
    // fails before an earlier script has run. Opening them before changing to the
    // data directory also keeps `-f` and `-r` paths relative to where the CLI was
    // started.
    let scripts = open_files(if args.repl_mode() {
        &args.rc
    } else {
        &args.file
    })?;
    if let Some(path) = &args.data_path {
        env::set_current_dir(Path::new(path))?;
    }

    let instrumented_registry = Arc::new(
        InstrumentedObjectStoreRegistry::new()
            .with_profile_mode(InstrumentedObjectStoreMode::Disabled),
    );
    let runtime_env = RuntimeEnvBuilder::new()
        .with_object_store_registry(Arc::clone(&instrumented_registry) as _)
        .build_arc()?;
    let state = SessionStateBuilder::new()
        .with_config(session_config(&args)?)
        .with_runtime_env(runtime_env)
        .with_default_features()
        .with_spark_features()
        .build();

    // Query files by their path or URL, such as `SELECT * FROM 'data.parquet'`, as
    // `datafusion-cli` does.
    let ctx = SessionContext::new_with_state(state).enable_url_table();
    ctx.refresh_catalogs().await?;
    ctx.register_catalog_list(Arc::new(DynamicObjectStoreCatalog::new(
        ctx.state().catalog_list().clone(),
        ctx.state_weak_ref(),
    )));

    let mut print_options = PrintOptions {
        format: args.format,
        quiet: args.quiet,
        maxrows: args.maxrows,
        color: args.color,
        instrumented_registry,
    };

    exec_files(&ctx, scripts, &print_options).await?;
    if args.repl_mode() {
        return exec::exec_from_repl(&ctx, &mut print_options)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)));
    }
    if !args.command.is_empty() {
        exec::exec_from_commands(&ctx, args.command.clone(), &print_options).await?;
    }
    Ok(())
}

/// Opens each file. Unlike `exec::exec_from_files`, which panics, a file that cannot
/// be opened is an error.
fn open_files(paths: &[String]) -> Result<Vec<File>> {
    paths
        .iter()
        .map(|path| {
            File::open(path).map_err(|e| {
                DataFusionError::Execution(format!("Cannot open file '{path}': {e}"))
            })
        })
        .collect()
}

/// Runs the SQL in each file in turn.
async fn exec_files(
    ctx: &SessionContext,
    files: Vec<File>,
    print_options: &PrintOptions,
) -> Result<()> {
    for file in files {
        exec::exec_from_lines(ctx, &mut BufReader::new(file), print_options).await?;
    }
    Ok(())
}

fn session_config(args: &Args) -> Result<SessionConfig> {
    // Read `DATAFUSION_*` environment variables, then apply the command line.
    let mut options = ConfigOptions::from_env()?;
    options.sql_parser.dialect = Dialect::Spark;
    options.execution.enable_ansi_mode = args.ansi;
    if let Some(batch_size) = args.batch_size {
        options.execution.batch_size = ConfigNonZeroUsize::try_new(batch_size)?;
    }
    // Defaults that `datafusion-cli` also uses, unless set in the environment.
    if env::var_os("DATAFUSION_EXPLAIN_FORMAT").is_none() {
        options.explain.format = ExplainFormat::Tree;
    }
    if env::var_os("DATAFUSION_FORMAT_NULL").is_none() {
        options.format.null = String::from("NULL");
    }

    let mut config = SessionConfig::from(options).with_information_schema(true);
    if args.reads_sql_from_stdin() {
        config = config.with_extension(Arc::new(StdinCarriesCommands));
    }
    Ok(config)
}

fn parse_valid_file(path: &str) -> Result<String, String> {
    // stdin is a pipe or a terminal, not a regular file.
    if is_stdin_location(path) || Path::new(path).is_file() {
        Ok(path.to_string())
    } else {
        Err(format!("Invalid file '{path}'"))
    }
}

fn parse_valid_data_dir(dir: &str) -> Result<String, String> {
    if Path::new(dir).is_dir() {
        Ok(dir.to_string())
    } else {
        Err(format!("Invalid data directory '{dir}'"))
    }
}

fn parse_batch_size(size: &str) -> Result<usize, String> {
    match size.parse::<usize>() {
        Ok(size) if size > 0 => Ok(size),
        _ => Err(format!("Invalid batch size '{size}'")),
    }
}

fn parse_command(command: &str) -> Result<String, String> {
    if command.is_empty() {
        Err("-c flag expects only non empty commands".to_string())
    } else {
        Ok(command.to_string())
    }
}
