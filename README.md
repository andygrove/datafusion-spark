<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# DataFusion Spark

[Apache DataFusion](https://datafusion.apache.org) expressions that emulate [Apache Spark](https://spark.apache.org)'s behavior.

This crate provides Spark-compatible scalar, aggregate, and table functions that can be registered with a DataFusion `SessionContext`.

## Quick start

```toml
[dependencies]
datafusion = "56"
datafusion-spark = { version = "56", features = ["core"] }
```

Call `with_spark_features` **after** `with_default_features`, since Spark functions replace DataFusion
functions with the same name:

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

See the [crate README](crates/spark/README.md) and the [API documentation](https://docs.rs/datafusion-spark)
for more.

## Repository layout

- `crates/spark` – the `datafusion-spark` crate
- `crates/sqllogictest` – SQL logic tests for the Spark functions
- `.ai/skills` – agent skills for working on this repository (see `AGENTS.md`)

## Building and testing

```shell
cargo build
cargo test
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for implementation and testing
guidelines. Before opening a PR, run:

```shell
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test -p datafusion-spark
cargo test -p datafusion-spark-sqllogictest
```

## License

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
