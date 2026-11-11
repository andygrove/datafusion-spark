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

## Repository layout

- `crates/spark` – the `datafusion-spark` crate
- `crates/sqllogictest` – SQL logic tests for the Spark functions

## Building and testing

```shell
cargo build
cargo test
```

## License

Licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).
