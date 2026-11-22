<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Contributing to datafusion-spark

Contributions are welcome. See also the
[DataFusion contributor guide](https://datafusion.apache.org/contributor-guide/).

## Implementation Guidelines

When implementing these functions, you can check if there are existing implementations
in the [Sail] or [Comet] projects first. If you do port functionality from these
sources, make sure to port over the corresponding tests too, to ensure correctness
and compatibility.

### `simplify()`

DataFusion functions allow you to implement `simplify()` which can let you rewrite
the function call during logical optimization, theoretically allowing you to avoid
implementing physical execution via `invoke_with_args()` if the rewrite is unconditional
(e.g. rewrite to an arithmetic operation).

**However, `invoke_with_args()` must always be implemented for functions in this
crate.** This is because downstream users such as Comet rely on DataFusion for physical
execution, and not logical planning/optimization. That means if a function doesn't
have a physical implementation (`invoke_with_args()`) it is not usable by Comet.

### Supported types

The functions in this crate need only support input types available to Spark; that
is, they do not need to handle unsigned types or types such as `Float16` or `Decimal64`.

### Input types, casts and ANSI mode

Spark casts a function's arguments before it runs the function, and an invalid value behaves
differently with ANSI mode on, off, and in the `try_*` functions. Prefer the shared helpers below
to per-function rules. They keep results and errors consistent, and their module docs have the
details.

- **Declared input types.** When the Spark expression is `ImplicitCastInputTypes`, declare its
  `inputTypes` with `SparkInputTypes` (`function/implicit_cast.rs`), which supplies `signature()`,
  `coerce_types()` and `with_updated_config()` with Spark's errors and ANSI rules. See `sha2`. It
  only chooses the target types, so the cast that then runs is not yet Spark's.
- **Checked input types.** When the expression uses `ExpectsInputTypes` without implicit casts,
  use `SparkInputTypes::check_only`. It shares the declared-type validation and Spark errors,
  accepts untyped NULLs, and unwraps dictionaries. Widen accepted inputs separately when the
  execution kernel needs a common representation. See `timestamp_micros`.
- **Common type of several arguments.** When Spark finds one type for several arguments, as
  `array_position` does, call the rules in `function/type_coercion.rs` (for example
  `tightest_common_type()`) from `coerce_types()`.
- **Casting a value.** To cast a value the way Spark's `CAST` does, call `spark_cast()`
  (`function/conversion/cast/`) and not Arrow's `cast`. The casts are being ported from Comet
  ([#88](https://github.com/apache/datafusion-spark/issues/88)). One that is not ported yet returns a
  "not implemented" error, because Arrow's result would differ from Spark's. Port the missing cast
  with its tests, or note in the PR that the function depends on it.
- **ANSI mode.** A function that depends on `spark.sql.ansi.enabled` should take an `EvalMode`
  (`Legacy`, `Ansi` or `Try`) rather than a `bool`, read with
  `EvalMode::from_config(&args.config_options)`. A `try_*` function then shares the implementation
  with `EvalMode::Try`, as `parse_url` and `try_parse_url` do.

[sail]: https://github.com/lakehq/sail
[comet]: https://github.com/apache/datafusion-comet

## Documentation Guidelines

Every function should have documentation that appears in generated function references
(`ScalarUDFImpl::documentation()`, or the aggregate/window equivalent). Write it once, using the
`#[user_doc]` attribute macro from `datafusion-macros` on the struct, which generates
`documentation()` and keeps the docs next to the type. The struct keeps only a short rustdoc
comment (what it is, plus a link to the Spark docs), so that facts are not written twice.

The most useful thing to document is **how the function differs from the DataFusion built-in of
the same name**, and its Spark edge-case behavior. Only document behavior that is covered by a
test.

```rust
use datafusion_macros::user_doc;

/// Spark-compatible `my_func` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#my_func>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "What it returns, and how it differs from the DataFusion built-in: types, \
        NULL handling, edge cases (out-of-range values, overflow, empty input), ANSI mode.",
    syntax_example = "my_func(expr[, optional])",
    alternative_syntax = "alias(expr)", // only if there are aliases
    sql_example = r#"```sql
> SELECT my_func('a');
result
```"#,
    argument(name = "expr", description = "What it is, accepted types, NULL behavior.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMyFunc { /* ... */ }

impl ScalarUDFImpl for SparkMyFunc {
    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
    // ...
}
```

See `sha2` (`crates/spark/src/function/hash/sha2.rs`) and `substring`
(`crates/spark/src/function/string/substring.rs`) for complete examples.

The function reference in [`docs/functions`](docs/functions/README.md) is generated from these
attributes, with one page per function category. After adding or changing a function's documentation, regenerate it and commit the
result (CI checks that it is up to date):

```shell
dev/update_function_docs.sh
```

The script also lists, on stderr, the functions that don't implement `documentation()` yet.

## Testing Guidelines

Prefer adding tests via SQLLogicTests where possible, see the [Spark SQLLogicTest README].
Resort to adding tests as Rust unit tests where it is impossible or difficult to
test via SLT. This is because direct invocation via Rust skips steps such as input
coercion, and is usually more verbose in the setup needed to pass data in (and
assert output data).

[spark sqllogictest readme]: crates/sqllogictest/test_files/README.md

## API Compatibility

`datafusion-spark` makes minor and patch releases between DataFusion major versions, so `main`
must not break its public API until it moves to the next DataFusion major version. For example, do
not remove or rename a public item, or change the signature of a public function. New functions and
new public items are fine, and so are changes to a function's results that make it match Spark. See
[Versioning](dev/release/README.md#versioning) for the full policy.

## Releases

The release process is documented in [dev/release/README.md](dev/release/README.md).
