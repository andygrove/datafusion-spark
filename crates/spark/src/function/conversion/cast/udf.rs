// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

use super::{SparkCastOptions, parse_spark_type, spark_cast};
use crate::{EvalMode, SparkTimeZone};

/// The time zone DataFusion uses when `datafusion.execution.time_zone` is not
/// set.
const DEFAULT_TIMEZONE: &str = "UTC";

/// `spark_cast(expr, '<type>')`: Spark's `CAST(expr AS <type>)`.
///
/// The target type is a string literal in Spark's SQL type syntax, such as
/// `'int'`, `'decimal(10, 2)'` or `'array<timestamp>'`. The cast follows
/// `datafusion.execution.enable_ansi_mode`, as Spark's follows
/// `spark.sql.ansi.enabled`, and uses `datafusion.execution.time_zone` as the
/// session time zone.
///
/// # References
/// - <https://spark.apache.org/docs/latest/api/sql/index.html#cast>
#[user_doc(
    doc_section(label = "Other Functions"),
    description = "Casts `expr` to `type` as Spark's `CAST(expr AS type)` does. The target type \
        is a string literal in Spark's SQL type syntax: BOOLEAN, TINYINT (BYTE), SMALLINT \
        (SHORT), INT (INTEGER), BIGINT (LONG), FLOAT (REAL), DOUBLE, DECIMAL(p, s) (DEC, \
        NUMERIC), STRING, CHAR(n), VARCHAR(n), BINARY, DATE, TIMESTAMP (TIMESTAMP_LTZ), \
        TIMESTAMP_NTZ, and ARRAY<...>, MAP<..., ...> and STRUCT<name: type, ...> of these. \
        Type names are case-insensitive. CHAR and VARCHAR require a length and cast to STRING. \
        INTERVAL, TIME, VARIANT, VOID and the geospatial types are not supported. The cast \
        follows `datafusion.execution.enable_ansi_mode`, as Spark's follows \
        `spark.sql.ansi.enabled`, and uses `datafusion.execution.time_zone` as the session \
        time zone. Casts are being ported from Comet \
        (https://github.com/apache/datafusion-spark/issues/88): a cast that Spark supports \
        but this function does not yet returns a \"not implemented yet\" error, and a cast \
        that Spark does not support returns a \"not supported\" error. Returns NULL if `expr` \
        is NULL.",
    syntax_example = "spark_cast(expr, type)",
    sql_example = r#"```sql
> SELECT spark_cast(true, 'int');
1
> SELECT spark_cast(0, 'timestamp');
1970-01-01T00:00:00Z
```"#,
    argument(name = "expr", description = "Expression to cast."),
    argument(
        name = "type",
        description = "The target type, as a string literal in Spark's SQL type syntax, such as 'int', 'decimal(10, 2)' or 'array<timestamp>'."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkCast {
    signature: Signature,
    /// The session time zone when the function was planned. A `TIMESTAMP`
    /// target is tagged with it.
    timezone: Arc<str>,
    /// `timezone` parsed with Spark's rules, or why it could not be. Only the
    /// casts that use the session time zone fail when it cannot be parsed.
    session_timezone: Result<SparkTimeZone, String>,
}

impl Default for SparkCast {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkCast {
    pub fn new() -> Self {
        Self::new_with_config(&ConfigOptions::default())
    }

    pub fn new_with_config(config: &ConfigOptions) -> Self {
        Self::with_session_timezone(
            config
                .execution
                .time_zone
                .as_ref()
                .map_or(DEFAULT_TIMEZONE, |tz| tz.as_str()),
        )
    }

    fn with_session_timezone(timezone: &str) -> Self {
        Self {
            // The second argument must be a string literal, which
            // `return_field_from_args` checks.
            signature: Signature::any(2, Volatility::Stable),
            timezone: timezone.into(),
            session_timezone: SparkTimeZone::parse(timezone).map_err(|e| {
                format!("the session time zone '{timezone}' is not supported: {e}")
            }),
        }
    }
}

impl ScalarUDFImpl for SparkCast {
    fn name(&self) -> &str {
        "spark_cast"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
        Some(ScalarUDF::from(Self::new_with_config(config)))
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let Some(Some(type_name)) = args
            .scalar_arguments
            .get(1)
            .and_then(|arg| arg.and_then(|arg| arg.try_as_str()))
        else {
            return plan_err!(
                "spark_cast requires a string literal target type, such as 'int'"
            );
        };
        let return_type = parse_spark_type(type_name, &self.timezone)?;
        Ok(Arc::new(Field::new(self.name(), return_type, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let options = SparkCastOptions::new(EvalMode::from_config(&args.config_options));
        let options = match &self.session_timezone {
            Ok(timezone) => options.with_timezone(timezone.clone()),
            Err(reason) => options.without_timezone_because(reason.clone()),
        };
        let [value, _] = take_function_args(self.name(), args.args)?;
        spark_cast(value, args.return_field.data_type(), &options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, ArrayRef, AsArray, Int32Array};
    use arrow::datatypes::{TimeUnit, TimestampMicrosecondType};
    use datafusion_common::ScalarValue;

    fn return_field(
        udf: &SparkCast,
        input: DataType,
        type_name: &str,
    ) -> Result<FieldRef> {
        let type_name = ScalarValue::Utf8(Some(type_name.to_string()));
        udf.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[
                Arc::new(Field::new("a", input, true)),
                Arc::new(Field::new("b", DataType::Utf8, false)),
            ],
            scalar_arguments: &[None, Some(&type_name)],
        })
    }

    fn invoke(
        udf: &SparkCast,
        config: ConfigOptions,
        input: ColumnarValue,
        type_name: &str,
    ) -> Result<ColumnarValue> {
        let return_field = return_field(udf, input.data_type(), type_name)?;
        udf.invoke_with_args(ScalarFunctionArgs {
            args: vec![
                input,
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(type_name.to_string()))),
            ],
            arg_fields: vec![],
            number_rows: 1,
            return_field,
            config_options: Arc::new(config),
        })
    }

    /// Invokes `spark_cast` planned and run with `config`.
    fn invoke_with(
        config: ConfigOptions,
        input: ColumnarValue,
        type_name: &str,
    ) -> Result<ColumnarValue> {
        invoke(
            &SparkCast::new_with_config(&config),
            config,
            input,
            type_name,
        )
    }

    fn config(time_zone: Option<&str>, ansi: bool) -> ConfigOptions {
        let mut config = ConfigOptions::default();
        config.execution.time_zone = time_zone.map(|tz| tz.parse().unwrap());
        config.execution.enable_ansi_mode = ansi;
        config
    }

    #[test]
    fn test_return_field_tags_timestamp_with_session_timezone() {
        let udf = SparkCast::new_with_config(&config(Some("America/Los_Angeles"), false));
        let field = return_field(&udf, DataType::Int64, "timestamp").unwrap();
        assert_eq!(
            field.data_type(),
            &DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("America/Los_Angeles".into())
            )
        );
        assert!(field.is_nullable());

        let udf = SparkCast::new_with_config(&config(None, false));
        let field = return_field(&udf, DataType::Int64, "timestamp").unwrap();
        assert_eq!(
            field.data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );
    }

    #[test]
    fn test_target_type_must_be_a_string_literal() {
        let udf = SparkCast::new();
        let error = udf
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[
                    Arc::new(Field::new("a", DataType::Int32, true)),
                    Arc::new(Field::new("b", DataType::Utf8, true)),
                ],
                scalar_arguments: &[None, None],
            })
            .unwrap_err();
        assert!(
            error.to_string().contains("string literal target type"),
            "{error}"
        );

        let error = return_field(&udf, DataType::Int32, "integer2").unwrap_err();
        assert!(
            error.to_string().contains("unknown type 'integer2'"),
            "{error}"
        );
    }

    #[test]
    fn test_invoke_array() {
        let input: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
        let result = invoke_with(
            config(Some("America/Los_Angeles"), false),
            ColumnarValue::Array(input),
            "timestamp",
        )
        .unwrap();
        let ColumnarValue::Array(result) = result else {
            panic!("expected an array, got {result:?}");
        };
        let result = result.as_primitive::<TimestampMicrosecondType>();
        assert_eq!(result.value(0), 1_000_000);
        assert!(result.is_null(1));
        assert_eq!(result.timezone(), Some("America/Los_Angeles"));
    }

    #[test]
    fn test_invoke_scalar() {
        let result = invoke_with(
            config(None, false),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(42))),
            "string",
        )
        .unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert_eq!(scalar, ScalarValue::Utf8(Some("42".to_string())));
    }

    #[test]
    fn test_unsupported_session_timezone_only_affects_casts_that_use_it() {
        // DataFusion accepts any time zone Arrow can parse, and Spark's parser
        // is stricter, so the session time zone can be one Spark rejects.
        let udf = SparkCast::with_session_timezone("+0530x");
        assert!(udf.session_timezone.is_err());
        let cast = |input: ScalarValue, type_name: &str| {
            invoke(
                &udf,
                ConfigOptions::default(),
                ColumnarValue::Scalar(input),
                type_name,
            )
        };
        assert!(cast(ScalarValue::Int32(Some(1)), "string").is_ok());

        let timestamp = ScalarValue::TimestampMicrosecond(Some(0), Some("UTC".into()));
        let error = cast(timestamp, "string").unwrap_err().to_string();
        assert!(error.contains("requires a session time zone"), "{error}");
        assert!(
            error.contains("the session time zone '+0530x' is not supported"),
            "{error}"
        );
    }

    #[test]
    fn test_invoke_reads_ansi_mode() {
        let nan = || ColumnarValue::Scalar(ScalarValue::Float64(Some(f64::NAN)));
        let result = invoke_with(config(None, false), nan(), "timestamp").unwrap();
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected a scalar, got {result:?}");
        };
        assert!(scalar.is_null());

        let error = invoke_with(config(None, true), nan(), "timestamp").unwrap_err();
        assert!(
            error.to_string().contains("[CAST_INVALID_INPUT]"),
            "{error}"
        );
    }
}
