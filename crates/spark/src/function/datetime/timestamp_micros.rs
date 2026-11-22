// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, FieldRef, TimeUnit};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::implicit_cast::{SparkInputType, SparkInputTypes};
use crate::function::nullability::{SparkNullability, spark_nullability};

/// Spark's `MicrosToTimestamp` is a null-intolerant `UnaryExpression` (an
/// `IntegralToTimestampBase`).
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

/// Spark-compatible `timestamp_micros` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#timestamp_micros>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the timestamp that is `microseconds` microseconds after \
        1970-01-01 00:00:00 UTC. Negative values precede the epoch. Unlike DataFusion's \
        `to_timestamp_micros`, only integer types whose full range fits in BIGINT are \
        accepted; floating-point, decimal and string inputs are rejected. The result has microsecond precision \
        and the session time zone (`datafusion.execution.time_zone`, default UTC). \
        The time zone affects display, not the stored microsecond count. Every BIGINT \
        value is valid, including its minimum and maximum. Returns NULL for NULL input.",
    syntax_example = "timestamp_micros(microseconds)",
    sql_example = r#"```sql
> SELECT timestamp_micros(0::BIGINT);
1970-01-01T00:00:00Z
> SELECT timestamp_micros(-1::BIGINT);
1969-12-31T23:59:59.999999Z
> SELECT timestamp_micros(NULL);
NULL
```"#,
    argument(
        name = "microseconds",
        description = "TINYINT, SMALLINT, INT or BIGINT count of microseconds since the UTC epoch. Arrow UInt8, UInt16 and UInt32 inputs are also accepted. UInt64 is rejected because its full range cannot fit in BIGINT. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkTimestampMicros {
    input_types: SparkInputTypes,
    timezone: Arc<str>,
}

impl Default for SparkTimestampMicros {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkTimestampMicros {
    pub fn new() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }

    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self {
            input_types: SparkInputTypes::check_only(
                vec![SparkInputType::Integral],
                Volatility::Stable,
            ),
            timezone: config
                .execution
                .time_zone
                .as_ref()
                .map_or("UTC", |tz| tz.as_str())
                .into(),
        }
    }
}

impl ScalarUDFImpl for SparkTimestampMicros {
    fn name(&self) -> &str {
        "timestamp_micros"
    }

    fn signature(&self) -> &Signature {
        self.input_types.signature()
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let types = self.input_types.coerce(self.name(), arg_types)?;
        if types[0] == DataType::UInt64 {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of \
                 `timestamp_micros` requires an integer type whose full range fits in \
                 BIGINT, however it has the type \"UInt64\". SQLSTATE: 42K09"
            );
        }
        Ok(vec![DataType::Int64])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [_] = take_function_args(self.name(), args.arg_fields)?;
        Ok(NULLABILITY.return_field(
            self.name(),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&self.timezone))),
            args.arg_fields,
        ))
    }

    spark_nullability!(NULLABILITY);

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::from(Self::new_with_config(config)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [microseconds] = take_function_args(self.name(), args.args)?;
        // Both types store an i64 count, so this does not scale the value or
        // convert it through a calendar representation (which has a smaller range).
        microseconds.cast_to(
            &DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::clone(&self.timezone))),
            None,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;

    #[test]
    fn return_field_tracks_session_timezone_and_input_nullability() -> Result<()> {
        for timezone in [None, Some("America/Los_Angeles")] {
            let mut config = ConfigOptions::default();
            config.execution.time_zone = timezone.map(str::parse).transpose()?;
            let function = SparkTimestampMicros::new()
                .with_updated_config(&config)
                .unwrap();
            for nullable in [false, true] {
                let input = Arc::new(Field::new("micros", DataType::Int64, nullable));
                let field = function.inner().return_field_from_args(ReturnFieldArgs {
                    arg_fields: &[input],
                    scalar_arguments: &[None],
                })?;
                assert_eq!(field.is_nullable(), nullable);
                assert_eq!(
                    field.data_type(),
                    &DataType::Timestamp(
                        TimeUnit::Microsecond,
                        Some(timezone.unwrap_or("UTC").into()),
                    ),
                );
            }
        }
        Ok(())
    }
}
