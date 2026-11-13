<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Running the Spark sqllogictests on Apache Spark

`pyspark_slt_runner.py` runs the `.slt` files in
[`crates/sqllogictest/test_files`](../test_files)
on Apache Spark 4.2 and reports every query where Spark's result differs from
the expected result in the file.

The expected results in those files come from DataFusion, and nothing else
checks them against Spark. A difference means that DataFusion and Spark
disagree, or that the test expects the wrong value. Running Spark is manual;
only the runner's unit tests run in CI.

## Setup

You need Python 3.10 or later and Java 17 or 21. Install PySpark 4.2.0 into a
virtual environment:

```shell
python3 -m venv /tmp/pyspark-4.2
/tmp/pyspark-4.2/bin/pip install -r crates/sqllogictest/scripts/requirements.txt
```

The runner ignores `SPARK_HOME`, so it always uses the Spark bundled with this
PySpark.

## Usage

From the repository root:

```shell
# All files
/tmp/pyspark-4.2/bin/python crates/sqllogictest/scripts/pyspark_slt_runner.py

# One file or directory
/tmp/pyspark-4.2/bin/python crates/sqllogictest/scripts/pyspark_slt_runner.py \
    crates/sqllogictest/test_files/math/abs.slt

# Also list skipped and passing records
/tmp/pyspark-4.2/bin/python crates/sqllogictest/scripts/pyspark_slt_runner.py --verbose
```

Each difference is printed with its `file:line`, the query, the expected result
and Spark's result. A summary of every outcome and the most common reasons for
skipping comes at the end. The runner exits with status 1 if it found any
differences, and with status 2 if it stopped early because Spark failed.

| Outcome                                             | Meaning                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------ |
| results differ                                      | Spark returns different rows                                             |
| Spark raises an error                               | The file expects a result, but Spark raises an error                     |
| DataFusion expects an error, Spark returns a result | The file expects an error, but Spark does not raise one                  |
| results match, column types differ                  | The rows match, but a column's type letter (`I`, `R`, `T`, ...) does not |
| same numbers, printed with a different scale        | For example `7` and `7.0`, usually a `DOUBLE` against a `DECIMAL`        |
| results differ only in floating-point digits        | The values differ by a relative error below 1e-12                        |
| same rows in a different order                      | Usually a query with no `ORDER BY`                                       |
| the runner failed                                   | A bug in this runner, not Spark behavior                                 |

A record that expects an error passes when Spark raises any error. The runner
does not compare the error message, so `pass` does not mean that Spark fails for
the same reason as DataFusion.

Spark reads a literal such as `1.5` as a `DECIMAL` and DataFusion reads it as a
`DOUBLE`, which explains many of the scale differences.

The runner collects at most 1000 rows per query, because some tests expect an
error from a query that Spark answers with billions of rows. It skips a
`stack()` of more than 1000 rows, such as `stack(2000000000, 1)`, because Spark
builds all of those rows at once and runs out of memory. If Spark fails anyway,
the runner skips the rest of that file and starts a new Spark session for the
next one.

## How queries are run

Each file starts with ANSI mode off and the session time zone set to UTC, which
are DataFusion's defaults. These settings in the files are passed on to Spark:

- `datafusion.execution.enable_ansi_mode` sets `spark.sql.ansi.enabled`
- `datafusion.execution.time_zone` sets `spark.sql.session.timeZone`
- `datafusion.spark.map_key_dedup_policy` sets `spark.sql.mapKeyDedupPolicy`

Spark 4.2 parses most of the SQL as written, including `::` casts. The runner
makes only these changes:

- `CREATE TABLE` becomes `CREATE TEMPORARY VIEW`, and `DROP TABLE` becomes
  `DROP VIEW`
- PostgreSQL type names that Spark does not accept are renamed, for example
  `FLOAT8` to `DOUBLE` and `TEXT` to `STRING`
- `column1`, `column2`, ... become `col1`, `col2`, ..., which is how Spark names
  `VALUES` columns
- `spark_cast(x, 'type')`, this crate's emulation of Spark's `CAST`, becomes
  `CAST(x AS type)`
- `arrow_cast(x, 'Type')` becomes `CAST(x AS type)` when the Arrow type has the
  same values as a Spark type. Types that differ only in their physical layout
  map to one Spark type: `Utf8`, `LargeUtf8` and `Utf8View` are all `STRING`,
  `LargeList(Int64)` and `FixedSizeList(3, Int64)` are `ARRAY<BIGINT>`, and a
  `Dictionary` is its value type. Arrow reads a number cast to a date or
  timestamp as a count of the type's unit, so those casts are translated only
  for literals: `arrow_cast(1703512245, 'Timestamp(s)')` becomes
  `timestamp_seconds(1703512245)`. In a UTC session, a timestamp without a
  time zone becomes Spark's `TIMESTAMP`, which holds the same values and is the
  type functions such as `unix_micros` accept. A date cast to an integer becomes
  `unix_date(x)`, because Arrow counts days since the epoch and Spark's `CAST`
  returns NULL.

These records are skipped and counted, with the reason, in the summary:

- queries Spark cannot parse, such as `[1, 2]` array literals
- queries that call a function Spark does not have, such as `arrow_typeof`
- `arrow_cast` to a type Spark does not have, such as an unsigned integer,
  `Float16`, or a nanosecond timestamp, and `arrow_cast` of a non-literal to a
  date, time or timestamp
- queries whose string literals contain a backslash. Spark reads the backslash
  as an escape character and DataFusion does not, so the two engines would see
  different strings.
- `DECIMAL` types with a precision above 38
- queries that use a table whose `CREATE TABLE` Spark could not run
- `EXPLAIN` and `DESCRIBE`, whose output differs between the engines
- results PySpark cannot return to Python, such as a map with array keys, and
  interval values, which this runner does not format
- records marked `skipif spark`, or `onlyif` another engine

## Values

Spark's results are printed the way DataFusion's sqllogictest engine prints
them, so they can be compared as text. For example, a `DOUBLE` keeps 15 decimal
places, a `DECIMAL` keeps its scale and an empty string is `(empty)`. A Spark
`TIMESTAMP` is printed in the session time zone. An expected timestamp with an
offset, such as `2024-07-15T00:00:00Z`, is converted to that time zone before
the comparison.

## Tests

The runner's parsing, translation and formatting have unit tests that need
neither Spark nor Java:

```shell
python3 -m unittest discover -s crates/sqllogictest/scripts
```
