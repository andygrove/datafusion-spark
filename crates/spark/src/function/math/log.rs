// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible logarithm functions: `ln`, `log10`, `log2`, `log1p` and `log`.
//!
//! Spark follows Hive here: an input at or below the function's asymptote
//! (0 for `ln`, `log10`, `log2` and both arguments of `log`, -1 for `log1p`)
//! returns NULL instead of the NaN or -Infinity that IEEE 754 gives. NaN is
//! not below the asymptote, so a NaN input returns NaN.
//!
//! DataFusion's built-ins return -Infinity or NaN for those inputs, and its
//! one-argument `log` is base 10 where Spark's is the natural logarithm.
//!
//! Spark computes the logarithms with `java.lang.StrictMath`, so these use the
//! fdlibm translation in [`super::fdlibm`] to return the same bits. Rust's
//! `f64::ln` differs in the last digit for some inputs.

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Float64Array};
use arrow::datatypes::{DataType, Float64Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use super::fdlibm;

/// Applies `f` to every value of a Float64 array, returning NULL where the
/// value is at or below `asymptote`, as Spark's `UnaryLogExpression` does.
fn unary_log(
    name: &str,
    args: &[ArrayRef],
    asymptote: f64,
    f: impl Fn(f64) -> f64,
) -> Result<ArrayRef> {
    let [arg] = take_function_args(name, args)?;
    let result: Float64Array = arg
        .as_primitive::<Float64Type>()
        .unary_opt(|x| if x <= asymptote { None } else { Some(f(x)) });
    Ok(Arc::new(result))
}

/// Defines a one-argument Spark logarithm UDF over DOUBLE.
macro_rules! unary_log_udf {
    ($struct:ident, $name:literal, $asymptote:expr, $f:expr) => {
        impl Default for $struct {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $struct {
            pub fn new() -> Self {
                Self {
                    signature: Signature::exact(
                        vec![DataType::Float64],
                        Volatility::Immutable,
                    ),
                }
            }
        }

        impl ScalarUDFImpl for $struct {
            fn name(&self) -> &str {
                $name
            }

            fn signature(&self) -> &Signature {
                &self.signature
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                Ok(DataType::Float64)
            }

            fn documentation(&self) -> Option<&Documentation> {
                self.doc()
            }

            fn invoke_with_args(
                &self,
                args: ScalarFunctionArgs,
            ) -> Result<ColumnarValue> {
                make_scalar_function(
                    |args: &[ArrayRef]| unary_log($name, args, $asymptote, $f),
                    vec![],
                )(&args.args)
            }
        }
    };
}

/// Spark-compatible `ln` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#ln>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the natural logarithm (base e) of `expr` as a DOUBLE. The argument is \
        cast to DOUBLE. Unlike DataFusion's built-in `ln`, which returns -Infinity for 0 and NaN for \
        a negative number, an input less than or equal to 0 returns NULL. Returns NULL if the input \
        is NULL, and NaN if it is NaN.",
    syntax_example = "ln(expr)",
    sql_example = r#"```sql
> SELECT ln(1);
0
> SELECT ln(0);
NULL
```"#,
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL, 0 or a negative number returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLn {
    signature: Signature,
}

unary_log_udf!(SparkLn, "ln", 0.0, fdlibm::log);

/// Spark-compatible `log10` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#log10>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the base 10 logarithm of `expr` as a DOUBLE. The argument is cast to \
        DOUBLE. Unlike DataFusion's built-in `log10`, which returns -Infinity for 0 and NaN for a \
        negative number, an input less than or equal to 0 returns NULL. Returns NULL if the input \
        is NULL, and NaN if it is NaN.",
    syntax_example = "log10(expr)",
    sql_example = r#"```sql
> SELECT log10(10);
1
> SELECT log10(-1);
NULL
```"#,
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL, 0 or a negative number returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLog10 {
    signature: Signature,
}

unary_log_udf!(SparkLog10, "log10", 0.0, fdlibm::log10);

/// Spark-compatible `log2` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#log2>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the base 2 logarithm of `expr` as a DOUBLE. The argument is cast to \
        DOUBLE. Like Spark, it is computed as `ln(expr) / ln(2)`, so it can differ from an exact \
        base 2 logarithm in the last digit. Unlike DataFusion's built-in `log2`, which returns \
        -Infinity for 0 and NaN for a negative number, an input less than or equal to 0 returns \
        NULL. Returns NULL if the input is NULL, and NaN if it is NaN.",
    syntax_example = "log2(expr)",
    sql_example = r#"```sql
> SELECT log2(2);
1
> SELECT log2(0);
NULL
```"#,
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL, 0 or a negative number returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLog2 {
    signature: Signature,
}

// Spark computes `StrictMath.log(x) / StrictMath.log(2)` rather than a true
// base 2 logarithm, which differs in the last bit for some inputs.
unary_log_udf!(SparkLog2, "log2", 0.0, |x: f64| {
    fdlibm::log(x) / fdlibm::log(2.0)
});

/// Spark-compatible `log1p` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#log1p>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns `ln(1 + expr)` as a DOUBLE, computed without the precision loss of \
        adding 1 to `expr` when `expr` is close to 0. DataFusion has no built-in `log1p`. The \
        argument is cast to DOUBLE. An input less than or equal to -1 returns NULL. Returns NULL \
        if the input is NULL, and NaN if it is NaN.",
    syntax_example = "log1p(expr)",
    sql_example = r#"```sql
> SELECT log1p(0);
0
> SELECT log1p(-1);
NULL
```"#,
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL or a number less than or equal to -1 returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLog1p {
    signature: Signature,
}

unary_log_udf!(SparkLog1p, "log1p", -1.0, fdlibm::log1p);

/// Spark-compatible `log` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#log>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the logarithm of `expr` in base `base` as a DOUBLE, computed as \
        `ln(expr) / ln(base)`. With one argument it returns the natural logarithm of `expr`, \
        where DataFusion's built-in `log` returns the base 10 logarithm. Both arguments are cast \
        to DOUBLE. Returns NULL if either argument is NULL or less than or equal to 0, otherwise \
        NaN if either is NaN. A base of 1 divides by 0, so it returns Infinity, -Infinity, or NaN \
        when `expr` is also 1.",
    syntax_example = "log([base, ]expr)",
    sql_example = r#"```sql
> SELECT log(10, 100);
2
> SELECT log(1);
0
> SELECT log(0, 10);
NULL
```"#,
    argument(
        name = "base",
        description = "Optional numeric base, cast to DOUBLE. Defaults to e. NULL, 0 or a negative number returns NULL."
    ),
    argument(
        name = "expr",
        description = "Numeric expression, cast to DOUBLE. NULL, 0 or a negative number returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLog {
    signature: Signature,
}

impl Default for SparkLog {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkLog {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Float64]),
                    TypeSignature::Exact(vec![DataType::Float64, DataType::Float64]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkLog {
    fn name(&self) -> &str {
        "log"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_log, vec![])(&args.args)
    }
}

fn spark_log(args: &[ArrayRef]) -> Result<ArrayRef> {
    match args {
        // Spark's one-argument `log` is `Logarithm(EulerNumber(), expr)`,
        // which its generated code evaluates as `StrictMath.log(expr)`.
        [_] => unary_log("log", args, 0.0, fdlibm::log),
        [base, value] => {
            let base = base.as_primitive::<Float64Type>();
            let value = value.as_primitive::<Float64Type>();
            let result: Float64Array = base
                .iter()
                .zip(value.iter())
                .map(|(base, value)| match (base, value) {
                    // NaN is not <= 0, so a NaN argument yields NaN
                    // unless the other argument is <= 0.
                    (Some(base), Some(value)) if base <= 0.0 || value <= 0.0 => None,
                    (Some(base), Some(value)) => {
                        Some(fdlibm::log(value) / fdlibm::log(base))
                    }
                    _ => None,
                })
                .collect();
            Ok(Arc::new(result))
        }
        _ => exec_err!("log expects 1 or 2 arguments, got {}", args.len()),
    }
}
