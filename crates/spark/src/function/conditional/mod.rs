// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark conditional functions, such as `if`.

use datafusion_expr::ScalarUDF;
use datafusion_functions::make_udf_function;
use std::sync::Arc;

mod r#if;
mod nanvl;

make_udf_function!(r#if::SparkIf, r#if);
make_udf_function!(nanvl::SparkNanvl, nanvl);

pub mod expr_fn {
    use datafusion_functions::export_functions;

    export_functions!((r#if, "If arg1 evaluates to true, then returns arg2; otherwise returns arg3", arg1 arg2 arg3));
    export_functions!((nanvl, "Returns expr1 if it is not NaN, or expr2 if expr1 is NaN", expr1 expr2));
}

pub fn functions() -> Vec<Arc<ScalarUDF>> {
    vec![r#if(), nanvl()]
}
