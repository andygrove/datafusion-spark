// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, FieldRef, Int64Type, TimeUnit};
use datafusion_common::cast::as_int64_array;
use datafusion_common::types::logical_date;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_doc::DocSection;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ExprSchemable, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `unix_date` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#unix_date>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the number of days since 1970-01-01, as an INT. \
        Dates before 1970-01-01 return a negative number. DataFusion has no built-in `unix_date`. \
        Returns NULL if the input is NULL. The input must be a DATE; strings are not implicitly cast.",
    syntax_example = "unix_date(date)",
    sql_example = r#"```sql
> SELECT unix_date(DATE '1970-01-02');
1
> SELECT unix_date(DATE '1900-01-02');
-25566
> SELECT unix_date(NULL::DATE);
NULL
```"#,
    argument(name = "date", description = "DATE expression. NULL returns NULL.")
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkUnixDate {
    signature: Signature,
}

impl Default for SparkUnixDate {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkUnixDate {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![Coercion::new_exact(TypeSignatureClass::Native(
                    logical_date(),
                ))],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkUnixDate {
    fn name(&self) -> &str {
        "unix_date"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields[0].is_nullable();
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, nullable)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("invoke_with_args should not be called on SparkUnixDate")
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [date] = take_function_args(self.name(), args)?;
        Ok(ExprSimplifyResult::Simplified(
            date.cast_to(&DataType::Date32, info.schema())?
                .cast_to(&DataType::Int32, info.schema())?,
        ))
    }
}

/// Spark-compatible `unix_micros`, `unix_millis` and `unix_seconds` expressions.
///
/// One struct serves all three functions, so the user docs are built per time unit
/// in `unix_timestamp_doc` rather than with `#[user_doc]`.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkUnixTimestamp {
    time_unit: TimeUnit,
    signature: Signature,
    name: &'static str,
}

impl SparkUnixTimestamp {
    pub fn new(name: &'static str, time_unit: TimeUnit) -> Self {
        Self {
            signature: Signature::coercible(
                vec![Coercion::new_exact(TypeSignatureClass::Timestamp)],
                Volatility::Immutable,
            ),
            time_unit,
            name,
        }
    }

    /// Returns the number of microseconds since epoch (1970-01-01 00:00:00 UTC) for the given timestamp.
    /// <https://spark.apache.org/docs/latest/api/sql/index.html#unix_micros>
    pub fn microseconds() -> Self {
        Self::new("unix_micros", TimeUnit::Microsecond)
    }

    /// Returns the number of milliseconds since epoch (1970-01-01 00:00:00 UTC) for the given timestamp.
    /// <https://spark.apache.org/docs/latest/api/sql/index.html#unix_millis>
    pub fn milliseconds() -> Self {
        Self::new("unix_millis", TimeUnit::Millisecond)
    }

    /// Returns the number of seconds since epoch (1970-01-01 00:00:00 UTC) for the given timestamp.
    /// <https://spark.apache.org/docs/latest/api/sql/index.html#unix_seconds>
    pub fn seconds() -> Self {
        Self::new("unix_seconds", TimeUnit::Second)
    }

    fn downscale_divisor(&self, input_type: &DataType) -> Option<i64> {
        match (input_type, self.time_unit) {
            (DataType::Timestamp(TimeUnit::Millisecond, _), TimeUnit::Second)
            | (DataType::Timestamp(TimeUnit::Microsecond, _), TimeUnit::Millisecond)
            | (DataType::Timestamp(TimeUnit::Nanosecond, _), TimeUnit::Microsecond) => {
                Some(1_000)
            }
            (DataType::Timestamp(TimeUnit::Microsecond, _), TimeUnit::Second)
            | (DataType::Timestamp(TimeUnit::Nanosecond, _), TimeUnit::Millisecond) => {
                Some(1_000_000)
            }
            (DataType::Timestamp(TimeUnit::Nanosecond, _), TimeUnit::Second) => {
                Some(1_000_000_000)
            }
            _ => None,
        }
    }
}

impl ScalarUDFImpl for SparkUnixTimestamp {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields[0].is_nullable();
        Ok(Arc::new(Field::new(self.name(), DataType::Int64, nullable)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static MICROS: LazyLock<Documentation> = LazyLock::new(|| {
            unix_timestamp_doc(
                "unix_micros",
                "microseconds",
                r#"```sql
> SELECT unix_micros(TIMESTAMP '1970-01-01 00:00:01');
1000000
> SELECT unix_micros(TIMESTAMP '1900-01-01 00:00:01');
-2208988799000000
> SELECT unix_micros(NULL::TIMESTAMP);
NULL
```"#,
            )
        });
        static MILLIS: LazyLock<Documentation> = LazyLock::new(|| {
            unix_timestamp_doc(
                "unix_millis",
                "milliseconds",
                r#"```sql
> SELECT unix_millis(TIMESTAMP '1970-01-01 00:00:01');
1000
> SELECT unix_millis(TIMESTAMP '1969-12-31 23:59:59.9985');
-2
> SELECT unix_millis(NULL::TIMESTAMP);
NULL
```"#,
            )
        });
        static SECONDS: LazyLock<Documentation> = LazyLock::new(|| {
            unix_timestamp_doc(
                "unix_seconds",
                "seconds",
                r#"```sql
> SELECT unix_seconds(TIMESTAMP '1970-01-01 00:00:01');
1
> SELECT unix_seconds(TIMESTAMP '1969-12-31 23:59:58.5');
-2
> SELECT unix_seconds(NULL::TIMESTAMP);
NULL
```"#,
            )
        });
        match self.time_unit {
            TimeUnit::Microsecond => Some(&MICROS),
            TimeUnit::Millisecond => Some(&MILLIS),
            TimeUnit::Second => Some(&SECONDS),
            TimeUnit::Nanosecond => None,
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [ts] = take_function_args(self.name(), args.args)?;
        let Some(divisor) = self.downscale_divisor(&ts.data_type()) else {
            return ts
                .cast_to(
                    &DataType::Timestamp(self.time_unit, Some("UTC".into())),
                    None,
                )?
                .cast_to(&DataType::Int64, None);
        };
        // Spark floors negative fractional timestamps, unlike Arrow's
        // truncating timestamp casts. The argument has already been evaluated once.
        match ts.cast_to(&DataType::Int64, None)? {
            ColumnarValue::Scalar(ScalarValue::Int64(ticks)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Int64(
                    ticks.map(|ticks| ticks.div_euclid(divisor)),
                )))
            }
            ColumnarValue::Array(ticks) => Ok(ColumnarValue::Array(Arc::new(
                as_int64_array(&ticks)?
                    .unary::<_, Int64Type>(|ticks| ticks.div_euclid(divisor)),
            ))),
            other => internal_err!("Expected Int64 ticks, got {other:?}"),
        }
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [ts] = take_function_args(self.name(), args)?;
        if self
            .downscale_divisor(&ts.get_type(info.schema())?)
            .is_some()
        {
            return Ok(ExprSimplifyResult::Original(vec![ts]));
        }
        Ok(ExprSimplifyResult::Simplified(
            ts.cast_to(
                &DataType::Timestamp(self.time_unit, Some("UTC".into())),
                info.schema(),
            )?
            .cast_to(&DataType::Int64, info.schema())?,
        ))
    }
}

/// Builds the user docs for `unix_micros`, `unix_millis` or `unix_seconds`.
fn unix_timestamp_doc(name: &str, unit: &str, sql_example: &str) -> Documentation {
    Documentation::builder(
        DocSection {
            include: true,
            label: "Time and Date Functions",
            description: None,
        },
        format!(
            "Returns the number of {unit} since 1970-01-01 00:00:00 UTC, as a BIGINT. \
            DataFusion has no built-in `{name}`. \
            A timestamp without a time zone is read as UTC, and a timestamp's time zone does not change the result. \
            Any finer part is rounded down (toward negative infinity, like Spark), \
            so timestamps before 1970 with a fractional part move one unit further from zero. \
            Returns NULL if the input is NULL. The input must be a TIMESTAMP; strings are not implicitly cast."
        ),
        format!("{name}(timestamp)"),
    )
    .with_sql_example(sql_example)
    .with_argument("timestamp", "TIMESTAMP expression. NULL returns NULL.")
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::{ScalarUDF, create_udf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn test_unix_timestamp_evaluates_volatile_input_once() -> Result<()> {
        for (function, unit, divisor) in [
            (SparkUnixTimestamp::seconds(), "Millisecond", 1_000),
            (SparkUnixTimestamp::milliseconds(), "Microsecond", 1_000),
            (SparkUnixTimestamp::microseconds(), "Nanosecond", 1_000),
            (SparkUnixTimestamp::seconds(), "Microsecond", 1_000_000),
            (SparkUnixTimestamp::milliseconds(), "Nanosecond", 1_000_000),
            (SparkUnixTimestamp::seconds(), "Nanosecond", 1_000_000_000),
        ] {
            let function = ScalarUDF::from(function);
            for array_input in [false, true] {
                let ctx = SessionContext::new();
                let calls = Arc::new(AtomicUsize::new(0));
                let udf_calls = Arc::clone(&calls);
                ctx.register_udf(create_udf(
                    "volatile_ticks",
                    vec![DataType::Int64],
                    DataType::Int64,
                    Volatility::Volatile,
                    Arc::new(move |args| {
                        let ticks =
                            if udf_calls.fetch_add(1, Ordering::SeqCst).is_multiple_of(2)
                            {
                                -1
                            } else {
                                divisor
                            };
                        Ok(match &args[0] {
                            ColumnarValue::Scalar(_) => {
                                ColumnarValue::Scalar(ScalarValue::Int64(Some(ticks)))
                            }
                            ColumnarValue::Array(array) => ColumnarValue::Array(
                                Arc::new(Int64Array::from(vec![ticks; array.len()])),
                            ),
                        })
                    }),
                ));
                ctx.register_udf(function.clone());
                let (input, from, expected) = if array_input {
                    (
                        "column1",
                        " FROM (VALUES (0), (1))",
                        vec![Some(-1), Some(-1)],
                    )
                } else {
                    ("0", "", vec![Some(-1)])
                };
                let query = format!(
                    "SELECT {}(arrow_cast(volatile_ticks({input}), \
                     'Timestamp({unit}, None)')) AS result{from}",
                    function.name()
                );
                let batches = ctx.sql(&query).await?.collect().await?;
                assert_eq!(calls.load(Ordering::SeqCst), 1, "{query}");
                let actual = batches
                    .iter()
                    .map(|batch| as_int64_array(batch.column(0)))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flat_map(|array| array.iter())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected, "{query}");
            }
        }
        Ok(())
    }
}
