// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::array::ArrayRef;
use arrow::compute::{DatePart, date_part};
use arrow::datatypes::DataType;
use datafusion_common::Result;
use datafusion_common::utils::take_function_args;
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Creates a signature for datetime extraction functions that accept timestamp types.
fn extract_signature() -> Signature {
    Signature::coercible(
        vec![Coercion::new_exact(TypeSignatureClass::Timestamp)],
        Volatility::Immutable,
    )
}

// -----------------------------------------------------------------------------
// SparkHour
// -----------------------------------------------------------------------------

/// Spark-compatible `hour` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#hour>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the hour of the day (0 to 23) of a timestamp, as an INT. \
        DataFusion has no built-in `hour`. \
        Returns NULL if the input is NULL. The input must be a TIMESTAMP; strings are not implicitly cast as they are in Spark ([#41](https://github.com/apache/datafusion-spark/issues/41)).",
    syntax_example = "hour(timestamp)",
    sql_example = r#"```sql
> SELECT hour(TIMESTAMP '2009-07-30 12:58:59');
12
> SELECT hour(NULL::TIMESTAMP);
NULL
```"#,
    argument(
        name = "timestamp",
        description = "TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkHour {
    signature: Signature,
}

impl Default for SparkHour {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkHour {
    pub fn new() -> Self {
        Self {
            signature: extract_signature(),
        }
    }
}

impl ScalarUDFImpl for SparkHour {
    fn name(&self) -> &str {
        "hour"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_hour, vec![])(&args.args)
    }
}

fn spark_hour(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [ts_arg] = take_function_args("hour", args)?;
    let result = date_part(ts_arg.as_ref(), DatePart::Hour)?;
    Ok(result)
}

// -----------------------------------------------------------------------------
// SparkMinute
// -----------------------------------------------------------------------------

/// Spark-compatible `minute` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#minute>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the minute of the hour (0 to 59) of a timestamp, as an INT. \
        DataFusion has no built-in `minute`. \
        Returns NULL if the input is NULL. The input must be a TIMESTAMP; strings are not implicitly cast as they are in Spark ([#41](https://github.com/apache/datafusion-spark/issues/41)).",
    syntax_example = "minute(timestamp)",
    sql_example = r#"```sql
> SELECT minute(TIMESTAMP '2009-07-30 12:58:59');
58
> SELECT minute(NULL::TIMESTAMP);
NULL
```"#,
    argument(
        name = "timestamp",
        description = "TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMinute {
    signature: Signature,
}

impl Default for SparkMinute {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMinute {
    pub fn new() -> Self {
        Self {
            signature: extract_signature(),
        }
    }
}

impl ScalarUDFImpl for SparkMinute {
    fn name(&self) -> &str {
        "minute"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_minute, vec![])(&args.args)
    }
}

fn spark_minute(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [ts_arg] = take_function_args("minute", args)?;
    let result = date_part(ts_arg.as_ref(), DatePart::Minute)?;
    Ok(result)
}

// -----------------------------------------------------------------------------
// SparkSecond
// -----------------------------------------------------------------------------

/// Spark-compatible `second` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#second>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the second of the minute (0 to 59) of a timestamp, as an INT. \
        Fractions of a second are dropped. \
        DataFusion has no built-in `second`. \
        Returns NULL if the input is NULL. The input must be a TIMESTAMP; strings are not implicitly cast as they are in Spark ([#41](https://github.com/apache/datafusion-spark/issues/41)).",
    syntax_example = "second(timestamp)",
    sql_example = r#"```sql
> SELECT second(TIMESTAMP '2009-07-30 12:58:59');
59
> SELECT second(NULL::TIMESTAMP);
NULL
```"#,
    argument(
        name = "timestamp",
        description = "TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkSecond {
    signature: Signature,
}

impl Default for SparkSecond {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkSecond {
    pub fn new() -> Self {
        Self {
            signature: extract_signature(),
        }
    }
}

impl ScalarUDFImpl for SparkSecond {
    fn name(&self) -> &str {
        "second"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        make_scalar_function(spark_second, vec![])(&args.args)
    }
}

fn spark_second(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [ts_arg] = take_function_args("second", args)?;
    let result = date_part(ts_arg.as_ref(), DatePart::Second)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int32Array, TimestampMicrosecondArray};
    use arrow::datatypes::TimeUnit;
    use std::sync::Arc;

    #[test]
    fn test_spark_hour() {
        // Create a timestamp array: 2024-01-15 14:30:45 UTC (in microseconds)
        // 14:30:45 -> hour = 14
        let ts_micros = 1_705_329_045_000_000_i64; // 2024-01-15 14:30:45 UTC
        let ts_array = TimestampMicrosecondArray::from(vec![Some(ts_micros), None]);
        let ts_array = Arc::new(ts_array) as ArrayRef;

        let result = spark_hour(&[ts_array]).unwrap();
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result.value(0), 14);
        assert!(result.is_null(1));
    }

    #[test]
    fn test_spark_minute() {
        // 14:30:45 -> minute = 30
        let ts_micros = 1_705_329_045_000_000_i64;
        let ts_array = TimestampMicrosecondArray::from(vec![Some(ts_micros), None]);
        let ts_array = Arc::new(ts_array) as ArrayRef;

        let result = spark_minute(&[ts_array]).unwrap();
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result.value(0), 30);
        assert!(result.is_null(1));
    }

    #[test]
    fn test_spark_second() {
        // 14:30:45 -> second = 45
        let ts_micros = 1_705_329_045_000_000_i64;
        let ts_array = TimestampMicrosecondArray::from(vec![Some(ts_micros), None]);
        let ts_array = Arc::new(ts_array) as ArrayRef;

        let result = spark_second(&[ts_array]).unwrap();
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();

        assert_eq!(result.value(0), 45);
        assert!(result.is_null(1));
    }

    #[test]
    fn test_hour_return_type() {
        let func = SparkHour::new();
        let result = func
            .return_type(&[DataType::Timestamp(TimeUnit::Microsecond, None)])
            .unwrap();
        assert_eq!(result, DataType::Int32);
    }

    #[test]
    fn test_minute_return_type() {
        let func = SparkMinute::new();
        let result = func
            .return_type(&[DataType::Timestamp(TimeUnit::Microsecond, None)])
            .unwrap();
        assert_eq!(result, DataType::Int32);
    }

    #[test]
    fn test_second_return_type() {
        let func = SparkSecond::new();
        let result = func
            .return_type(&[DataType::Timestamp(TimeUnit::Microsecond, None)])
            .unwrap();
        assert_eq!(result, DataType::Int32);
    }
}
