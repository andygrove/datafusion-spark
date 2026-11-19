<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Agent Guidelines for datafusion-spark

DataFusion expressions that emulate Apache Spark's behavior.

- `crates/spark` – the `datafusion-spark` crate
- `crates/sqllogictest` – `.slt` tests for the Spark functions
- `crates/docgen` – generates the `docs/functions` reference from `#[user_doc]` (run `dev/update_function_docs.sh`)

Before opening a PR, run `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`,
`cargo test -p datafusion-spark` and `cargo test -p datafusion-spark-sqllogictest`.
If you changed any `#[user_doc]` attribute, also run `dev/update_function_docs.sh` and commit the changes under `docs/functions`.

## Agent Skills

Repository-specific agent skills live under `.ai/skills/<skill-name>/SKILL.md`. Check that directory for
applicable skills before working on a task.
