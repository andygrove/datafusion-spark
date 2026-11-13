// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, TimeUnit};
use datafusion_common::types::{NativeType, logical_string};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err, plan_err};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ExprSchemable, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `date_trunc` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_trunc>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns `ts` truncated to the unit given by `fmt`, with the same type as `ts`. \
        Unlike DataFusion's `date_trunc`, it also accepts Spark's unit aliases `yyyy` and `yy` for year, \
        `mon` and `mm` for month and `dd` for day, and it truncates a timestamp with a time zone in the session time zone \
        (`datafusion.execution.time_zone`) rather than the timestamp's own time zone. A string `ts` is cast to a timestamp. \
        Returns NULL if `ts` is NULL. `fmt` must be a constant string: a NULL `fmt` is a planning error and an unsupported \
        `fmt` is an execution error, where Spark returns NULL for both.",
    syntax_example = "date_trunc(fmt, ts)",
    sql_example = r#"```sql
> SELECT date_trunc('YEAR', '2015-03-05T09:32:05.123456'::timestamp);
2015-01-01T00:00:00
> SELECT date_trunc('MM', '2015-03-05T09:32:05.123456'::timestamp);
2015-03-01T00:00:00
> SELECT date_trunc('HOUR', '2015-03-05T09:32:05.123456'::timestamp);
2015-03-05T09:00:00
> SELECT date_trunc('MILLISECOND', '2015-03-05T09:32:05.123456'::timestamp);
2015-03-05T09:32:05.123
```"#,
    argument(
        name = "fmt",
        description = "Constant string unit, case-insensitive: `year` (`yyyy`, `yy`), `quarter`, `month` (`mon`, `mm`), `week` (truncates to Monday), `day` (`dd`), `hour`, `minute`, `second`, `millisecond` or `microsecond`."
    ),
    argument(
        name = "ts",
        description = "Timestamp expression to truncate. A string is cast to a timestamp. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDateTrunc {
    signature: Signature,
}

impl Default for SparkDateTrunc {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDateTrunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                    Coercion::new_implicit(
                        TypeSignatureClass::Timestamp,
                        vec![TypeSignatureClass::Native(logical_string())],
                        NativeType::Timestamp(TimeUnit::Microsecond, None),
                    ),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkDateTrunc {
    fn name(&self) -> &str {
        "date_trunc"
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());

        Ok(Arc::new(Field::new(
            self.name(),
            args.arg_fields[1].data_type().clone(),
            nullable,
        )))
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!(
            "spark date_trunc should have been simplified to standard date_trunc"
        )
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [fmt_expr, ts_expr] = take_function_args(self.name(), args)?;

        let fmt = match fmt_expr.as_literal() {
            Some(ScalarValue::Utf8(Some(v)))
            | Some(ScalarValue::Utf8View(Some(v)))
            | Some(ScalarValue::LargeUtf8(Some(v))) => v.to_lowercase(),
            _ => {
                return plan_err!(
                    "First argument of `DATE_TRUNC` must be non-null scalar Utf8"
                );
            }
        };

        // Map Spark-specific fmt aliases to datafusion ones
        let fmt = match fmt.as_str() {
            "yy" | "yyyy" => "year",
            "mm" | "mon" => "month",
            "dd" => "day",
            other => other,
        };

        let session_tz = info.config_options().execution.time_zone.clone();
        let ts_type = ts_expr.get_type(info.schema())?;

        // Spark interprets timestamps in the session timezone before truncating,
        // then returns a timestamp at microsecond precision.
        // See: https://github.com/apache/spark/blob/f310f4fcc95580a6824bc7d22b76006f79b8804a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/util/DateTimeUtils.scala#L492
        //
        // For sub-second truncations (second, millisecond, microsecond), timezone
        // adjustment is unnecessary since timezone offsets are whole seconds.
        let ts_expr = match (&ts_type, fmt) {
            // Sub-second truncations don't need timezone adjustment
            (_, "second" | "millisecond" | "microsecond") => ts_expr,

            // convert to session timezone, strip timezone and convert back to original timezone
            (DataType::Timestamp(unit, tz), _) => {
                let ts_expr = match &session_tz {
                    Some(session_tz) => ts_expr.cast_to(
                        &DataType::Timestamp(
                            TimeUnit::Microsecond,
                            Some(Arc::from(session_tz.as_str())),
                        ),
                        info.schema(),
                    )?,
                    None => ts_expr,
                };
                Expr::ScalarFunction(ScalarFunction::new_udf(
                    datafusion_functions::datetime::to_local_time(),
                    vec![ts_expr],
                ))
                .cast_to(&DataType::Timestamp(*unit, tz.clone()), info.schema())?
            }

            _ => {
                return plan_err!(
                    "Second argument of `DATE_TRUNC` must be Timestamp, got {}",
                    ts_type
                );
            }
        };

        let fmt_expr = Expr::Literal(ScalarValue::new_utf8(fmt), None);

        Ok(ExprSimplifyResult::Simplified(Expr::ScalarFunction(
            ScalarFunction::new_udf(
                datafusion_functions::datetime::date_trunc(),
                vec![fmt_expr, ts_expr],
            ),
        )))
    }
}
