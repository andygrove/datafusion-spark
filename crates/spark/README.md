<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Apache DataFusion Spark-compatible Expressions

[![Crates.io](https://img.shields.io/crates/v/datafusion-spark.svg)](https://crates.io/crates/datafusion-spark)
[![docs.rs](https://docs.rs/datafusion-spark/badge.svg)](https://docs.rs/datafusion-spark)

[Apache DataFusion] is an extensible query execution framework, written in Rust, that uses [Apache Arrow] as its in-memory format.

`datafusion-spark` provides [Apache Spark] compatible expressions for use with DataFusion: scalar,
aggregate, window and table functions that follow Spark's semantics, plus a Spark-aware expression
planner.

DataFusion's built-in functions follow PostgreSQL / DuckDB conventions, so a function with the same
name can behave differently in Spark (NULL handling, overflow, string indexing, rounding, ANSI mode
errors, ...). This crate is useful when you are:

- building an engine on DataFusion that must produce the same results as Spark (for example
  [Apache DataFusion Comet]), or
- running SQL migrated from Spark on DataFusion.

Not every Spark function is implemented yet. Progress is tracked in the
[`datafusion-spark` epic]; see the [API documentation] for the functions available today.

## Quick start

Add the crate with the `core` feature, which provides the `SessionStateBuilderSpark` helper:

```toml
[dependencies]
datafusion = "56"
datafusion-spark = { version = "56", features = ["core"] }
```

Add the `sql` feature as well to enable the Spark planner rewrites. That is what gives operators such
as `/` their Spark semantics (both sides cast to `DOUBLE`, and a zero divisor returns `NULL`):

```toml
datafusion-spark = { version = "56", features = ["core", "sql"] }
```

Enable Spark features on a session. Call `with_spark_features` **after** `with_default_features`, since
Spark functions replace DataFusion functions with the same name:

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use datafusion_spark::SessionStateBuilderSpark;

let state = SessionStateBuilder::new()
    .with_default_features()
    .with_spark_features()
    .build();
let ctx = SessionContext::new_with_state(state);

// `sha2` now has Spark semantics
let df = ctx.sql("SELECT sha2('Spark', 256)").await?;
```

Without the `core` feature the crate only depends on DataFusion's building-block crates. You can then
register functions with any `FunctionRegistry` using `datafusion_spark::register_all`, or pick
individual functions from the `datafusion_spark::function::*` packages. Each package also has an
`expr_fn` module for building expressions from Rust:

```rust
use datafusion_expr::{col, lit};
use datafusion_spark::expr_fn::sha2;

let expr = sha2(col("my_data"), lit(256));
```

## Using the DataFusion CLI

Pass `--spark` to `datafusion-cli` to enable the Spark SQL dialect and register
the Spark-compatible scalar, aggregate, and window functions for the session:

```shell
datafusion-cli --spark
```

The option is disabled by default because Spark-compatible functions can
override DataFusion functions with the same name.

## Things to know

- **Spark functions override built-ins** with the same name for the session they are registered in.
- **ANSI mode**: functions whose behavior depends on it honor `datafusion.execution.enable_ansi_mode`.
- **Input types** are limited to those Spark supports (no unsigned integers, `Float16`, `Decimal64`).
- **Explicit casts** are recommended in SQL, since DataFusion and Spark infer literal types differently.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/apache/datafusion-spark/blob/main/CONTRIBUTING.md)
and the [DataFusion contributor guide](https://datafusion.apache.org/contributor-guide/).

[apache arrow]: https://arrow.apache.org/
[apache datafusion]: https://datafusion.apache.org/
[apache spark]: https://spark.apache.org/
[apache datafusion comet]: https://datafusion.apache.org/comet/
[api documentation]: https://docs.rs/datafusion-spark
[`datafusion-spark` epic]: https://github.com/apache/datafusion/issues/15914
