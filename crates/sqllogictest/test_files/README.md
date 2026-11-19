<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Spark Test Files

This directory contains test files for the `spark` test suite.

## RoadMap

Implementing the `datafusion-spark` compatible functions project is still a work in progress.
Many of the tests in this directory are commented out and are waiting for help with implementation.

For more information please see:

- [The `datafusion-spark` Epic](https://github.com/apache/datafusion/issues/15914)
- [Spark Test Generation Script] (https://github.com/apache/datafusion/pull/16409#issuecomment-2972618052)

## Testing Guide

When testing Spark functions:

- Functions must be tested on both `Scalar` and `Array` inputs
- Test cases should only contain `SELECT` statements with the function being tested
- Add explicit casts to input values to ensure the correct data type is used (e.g., `0::INT`)
  - Explicit casting is necessary because DataFusion and Spark do not infer data types in the same way
- If the Spark built-in function under test behaves differently in ANSI SQL mode, please wrap your test cases like this example:

```sql
statement ok
set datafusion.execution.enable_ansi_mode = true;

# Functions under test
select abs((-128)::TINYINT)

statement ok
set datafusion.execution.enable_ansi_mode = false;
```

Run `cargo run -p datafusion-spark-cli` to explore Spark-compatible functions
interactively. The CLI enables the Spark SQL dialect and registers the
Spark-compatible function library. Unlike these test files, it starts with
ANSI mode on, as Spark 4 does. Pass `--ansi false` to turn it off.

### Finding Test Cases

To verify and compare function behavior at a minimum, you can refer to the following documentation sources:

1. Databricks SQL Function Reference:
   https://docs.databricks.com/aws/en/sql/language-manual/functions/NAME
2. Apache Spark SQL Function Reference:
   https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.NAME.html
3. PySpark SQL Function Reference:
   https://spark.apache.org/docs/latest/api/sql/#NAME

**Note:** Replace `NAME` in each URL with the actual function name (e.g., for the `ASCII` function, use `ascii` instead
of `NAME`).

### Scalar Example:

```sql
SELECT expm1(0::INT);
```

### Array Example:

```sql
SELECT expm1(a) FROM (VALUES (0::INT), (1::INT)) AS t(a);
```

## Simplify and evaluate modes

`cargo test -p datafusion-spark-sqllogictest` runs every file twice:

- `simplify` uses the default optimizer, so a call whose arguments are all literals,
  such as `SELECT expm1(0::INT)`, is constant-folded while planning. This calls the
  function's `simplify` (if it has one) and evaluates it on a one-row batch.
- `evaluate` replaces the `simplify_expressions` optimizer rule, so a call that reaches
  the physical plan runs through `invoke_with_args` during execution, as it does in
  engines such as Comet that build physical expressions directly. DataFusion functions
  that only work as a rewrite, such as `arrow_cast`, are still rewritten.

`evaluate` mode only changes the optimizer. SQL planning still runs first and can
evaluate or simplify some expressions before the optimizer sees them, for example the
arguments of a table function. A call in a projection or filter over rows, such as
`SELECT expm1(a) FROM (VALUES (0::INT), (1::INT)) AS t(a)`, is the reliable way to test
`invoke_with_args` directly.

Every function must work in both modes: `invoke_with_args` is required, and `simplify`
is an optional optimization. Pass `--mode=simplify` or `--mode=evaluate` to run one mode,
for example `cargo test -p datafusion-spark-sqllogictest -- math/expm1 --mode=evaluate`.

A record that cannot run in `evaluate` mode yet is marked `skipif evaluate`, with a
comment that links the issue. Remove the marks when the issue is fixed.

## Checking the expected results against Spark

The expected results in these files come from DataFusion. To compare them with
Apache Spark, see the [PySpark runner](../scripts/README.md).
