// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::{ArrayRef, AsArray, BooleanArray};
use arrow::compute::kernels::zip::zip;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float32Type, Float64Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err, plan_err};
use datafusion_expr::expr::{Cast, TryCast};
use datafusion_expr::expr_fn::when;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ExprSchemable, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_functions::math::expr_fn::isnan;
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark-compatible `nanvl` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#nanvl>
#[user_doc(
    doc_section(label = "Conditional Functions"),
    description = "Returns `expr1` if it is not NaN, or `expr2` if `expr1` is NaN. Unlike \
        DataFusion's built-in `nanvl`, which evaluates both arguments eagerly, this function \
        only evaluates `expr2` when `expr1` is NaN, matching Spark's short-circuiting behavior. \
        If `expr1` is NULL the result is NULL, even when `expr2` is not NULL; a NULL `expr2` \
        only matters when `expr1` is NaN. Both arguments are implicitly cast to DOUBLE (integer, \
        DECIMAL and STRING arguments included); if both arguments are FLOAT the result is FLOAT, \
        otherwise the result is DOUBLE. +/-Infinity and -0.0 are not NaN and are returned \
        unchanged, with the sign of zero preserved.",
    syntax_example = "nanvl(expr1, expr2)",
    sql_example = r#"```sql
> SELECT nanvl(cast('NaN' as double), 123);
123
> SELECT nanvl(5.0, CAST(NULL AS DOUBLE));
5
> SELECT nanvl(CAST(NULL AS DOUBLE), 5.0);
NULL
> SELECT nanvl(CAST('NaN' AS DOUBLE), CAST('NaN' AS DOUBLE));
NaN
```"#,
    argument(
        name = "expr1",
        description = "Numeric expression, cast to DOUBLE (or FLOAT if both arguments are \
            FLOAT). NULL returns NULL regardless of expr2."
    ),
    argument(
        name = "expr2",
        description = "Numeric expression, cast to match expr1. Only evaluated, and only \
            matters, when expr1 is NaN."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkNanvl {
    signature: Signature,
}

impl Default for SparkNanvl {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkNanvl {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkNanvl {
    fn name(&self) -> &str {
        "nanvl"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [t1, t2] = take_function_args("nanvl", arg_types)?;

        for t in [t1, t2] {
            if !is_nanvl_coercible(t) {
                return plan_err!(
                    "nanvl does not support the input type {t}; expected a \
                     numeric, decimal, string or null type"
                );
            }
        }

        if is_string_type(t1) || is_string_type(t2) {
            // A STRING argument's numeric parse can fail at runtime, and
            // whether that failure is NULL (legacy) or an error (ANSI)
            // depends on `enable_ansi_mode`. Letting the analyzer insert
            // its own implicit Cast here would always use a strict cast
            // regardless of ANSI mode, so a STRING argument is left as-is
            // here and cast manually, ANSI-aware, in `simplify()`. Spark
            // resolves a STRING argument against DOUBLE, never FLOAT, so
            // the other side is coerced to DOUBLE too (a numeric -> DOUBLE
            // cast never fails, so the analyzer can still insert that one
            // safely).
            let resolve = |t: &DataType| {
                if is_string_type(t) {
                    t.clone()
                } else {
                    DataType::Float64
                }
            };
            return Ok(vec![resolve(t1), resolve(t2)]);
        }

        // Spark coerces a NULL *left* argument straight to DOUBLE
        // (TypeCollection(DoubleType, FloatType)'s default for an untyped
        // NULL), but a NULL *right* argument instead adopts whatever type
        // the left argument resolves to. The rule is not symmetric:
        // nanvl(NULL, FLOAT) -> DOUBLE, but nanvl(FLOAT, NULL) -> FLOAT.
        let is_float32 = |dt: &DataType| matches!(dt, DataType::Float32);

        let target = match (t1, t2) {
            (DataType::Null, _) => DataType::Float64,
            (t1, DataType::Null) if is_float32(t1) => DataType::Float32,
            (t1, t2) if is_float32(t1) && is_float32(t2) => DataType::Float32,
            _ => DataType::Float64,
        };

        Ok(vec![target.clone(), target])
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        let [t1, _t2] = take_function_args("nanvl", arg_types)?;
        if is_string_type(t1) {
            // A STRING argument stays unresolved through coerce_types (see
            // above), but it always produces a DOUBLE result.
            Ok(DataType::Float64)
        } else {
            Ok(t1.clone())
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [expr1, expr2] = take_function_args("nanvl", args)?;

        // A STRING argument is left unresolved by `coerce_types` (see its
        // comment) so it can be cast here instead of by the analyzer's own
        // always-strict implicit Cast. Spark's semantics are ANSI-aware: a
        // bad numeric string is NULL in legacy mode and an error under
        // ANSI. Wrapping it in Cast/TryCast keeps that cast inside the
        // expression tree, where DataFusion's CASE evaluates it lazily -
        // a string expr2 is only cast (and can only fail) on rows where
        // expr1 is actually NaN, matching Spark. This must run before the
        // volatility check below: `invoke_with_args` only ever handles
        // Float32/Float64 arrays, so a string argument must already be
        // cast to Float64 by the time a volatile expr1 falls back to it.
        let enable_ansi_mode = info.config_options().execution.enable_ansi_mode;
        let to_float64 = |e: Expr| -> Result<Expr> {
            if !is_string_type(&e.get_type(info.schema())?) {
                return Ok(e);
            }
            Ok(if enable_ansi_mode {
                Expr::Cast(Cast::new(Box::new(e), DataType::Float64))
            } else {
                Expr::TryCast(TryCast::new(Box::new(e), DataType::Float64))
            })
        };
        let expr1 = to_float64(expr1)?;
        let expr2 = to_float64(expr2)?;

        // A volatile expr1 (e.g. `rand()`) must be evaluated exactly once.
        // The CASE rewrite below references expr1 twice - once in the
        // condition and once in the ELSE branch - which would evaluate it
        // separately in each place. Skip the rewrite for volatile input and
        // fall back to `invoke_with_args`, which evaluates each argument
        // array exactly once. The string cast above has already run, so
        // invoke_with_args only ever sees Float32/Float64 arrays here.
        //
        // Note this fallback also makes expr2 eager (invoke_with_args
        // evaluates both arguments up front), so e.g.
        // `nanvl(random(), 1 / 0)` under ANSI mode raises here even though
        // Spark would not, since Spark only evaluates `1 / 0` on rows where
        // `random()` is NaN (which never happens). Accepted as an edge
        // case: a volatile expr1 paired with a fallible expr2 is unusual.
        if expr1.is_volatile() {
            return Ok(ExprSimplifyResult::Original(vec![expr1, expr2]));
        }

        // Arrow/DataFusion comparison kernels use IEEE-754 totalOrder
        // semantics, under which NaN compares equal to itself, so
        // `expr1 != expr1` is never true for a NaN value coming from a
        // column - it only appears to work for literal NaNs that the
        // simplifier constant-folds ahead of time. Use `isnan` instead,
        // which inspects the actual bit pattern.
        let is_nan = isnan(expr1.clone());
        let case_expr = when(is_nan, expr2).otherwise(expr1)?;
        Ok(ExprSimplifyResult::Simplified(case_expr))
    }
    // This eager path is used only for physical execution that bypasses
    // the logical optimizer entirely (e.g. Comet), or when `simplify()`
    // falls back to `Original` for a volatile expr1 (the string cast has
    // already been applied in that case, so this only ever needs to
    // handle Float32/Float64). `return_type` reports Float64 for a Utf8
    // input type so that a plan which reaches here directly (bypassing
    // `simplify()`) still type-checks; this explicit check turns a
    // mismatch at that point into a clear error instead of letting Arrow's
    // `zip` kernel fail on a raw Utf8 array with a confusing message.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        for arg in &args.args {
            if !matches!(arg.data_type(), DataType::Float32 | DataType::Float64) {
                return exec_err!(
                    "nanvl expects Float32 or Float64 arguments at this point, got {}",
                    arg.data_type()
                );
            }
        }
        match args.args[0].data_type() {
            DataType::Float32 => {
                make_scalar_function(spark_nanvl_f32, vec![])(&args.args)
            }
            DataType::Float64 => {
                make_scalar_function(spark_nanvl_f64, vec![])(&args.args)
            }
            other => exec_err!("Unsupported data type {other} for function nanvl"),
        }
    }
}

/// Spark's `ImplicitCastInputTypes` for `nanvl` accepts
/// `TypeCollection(DoubleType, FloatType)`, which numeric types (integers,
/// FLOAT, DOUBLE, DECIMAL), STRING and NULL can all implicitly cast into.
/// Other types, such as BOOLEAN, cannot, and must be rejected rather than
/// silently accepted just because Arrow happens to support a cast.
fn is_nanvl_coercible(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Null
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
    )
}

fn is_string_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Builds the boolean mask "is expr1 NaN" (never true for a null expr1,
/// since a null must fall through to the ELSE branch unchanged) and
/// selects `expr2` where it is true, `expr1` otherwise - vectorised via
/// Arrow's `zip` kernel. Shared by the f32 and f64 entry points below so
/// the actual selection logic exists exactly once.
fn nanvl_generic<T: ArrowPrimitiveType>(
    args: &[ArrayRef],
    is_nan: impl Fn(T::Native) -> bool,
) -> Result<ArrayRef> {
    let [expr1, expr2] = take_function_args("nanvl", args)?;
    let expr1_prim = expr1.as_primitive::<T>();
    let mask: BooleanArray = expr1_prim
        .iter()
        .map(|opt| matches!(opt, Some(v) if is_nan(v)))
        .collect();
    Ok(zip(&mask, expr2, expr1)?)
}

fn spark_nanvl_f64(args: &[ArrayRef]) -> Result<ArrayRef> {
    nanvl_generic::<Float64Type>(args, f64::is_nan)
}

fn spark_nanvl_f32(args: &[ArrayRef]) -> Result<ArrayRef> {
    nanvl_generic::<Float32Type>(args, f32::is_nan)
}
