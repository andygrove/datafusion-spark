// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Array, Date32Array};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::cast::as_int32_array;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use super::date_part_name::int32_coercion;

/// Spark-compatible `date_from_unix_date` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_from_unix_date>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the date that is `days` days after 1970-01-01. This is the inverse of `unix_date`. \
        DataFusion has no built-in equivalent. Negative values give dates before the epoch. \
        Any INT is a valid date. Other numeric and string inputs are cast to INT first, which \
        truncates fractions and raises an error if the value is out of the INT range or is \
        not a valid number. These errors match Spark with ANSI mode enabled. With ANSI mode \
        disabled, Spark wraps an out-of-range BIGINT and returns NULL for a malformed string, \
        while this function still errors (see https://github.com/apache/datafusion-spark/issues/88). \
        Returns NULL if the input is NULL.",
    syntax_example = "date_from_unix_date(days)",
    sql_example = r#"```sql
> SELECT date_from_unix_date(1000);
1972-09-27
> SELECT date_from_unix_date(-1);
1969-12-31
> SELECT date_from_unix_date(NULL);
NULL
```"#,
    argument(
        name = "days",
        description = "Number of days since 1970-01-01, as an INT. Other integer, floating-point, decimal and string inputs are cast to INT. BOOLEAN is rejected. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDateFromUnixDate {
    signature: Signature,
}

impl Default for SparkDateFromUnixDate {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDateFromUnixDate {
    pub fn new() -> Self {
        // Spark declares the argument as INT.
        Self {
            signature: Signature::coercible(
                vec![int32_coercion()],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkDateFromUnixDate {
    fn name(&self) -> &str {
        "date_from_unix_date"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields[0].is_nullable();
        Ok(Arc::new(Field::new(
            self.name(),
            DataType::Date32,
            nullable,
        )))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [days] = take_function_args(self.name(), args.args)?;
        match days {
            // Date32 is Int32 days since the epoch, so the buffer is reused as is.
            ColumnarValue::Array(array) => {
                let days = as_int32_array(&array)?;
                Ok(ColumnarValue::Array(Arc::new(Date32Array::new(
                    days.values().clone(),
                    days.nulls().cloned(),
                ))))
            }
            ColumnarValue::Scalar(ScalarValue::Int32(days)) => {
                Ok(ColumnarValue::Scalar(ScalarValue::Date32(days)))
            }
            ColumnarValue::Scalar(other) => {
                internal_err!(
                    "date_from_unix_date expects INT, got {}",
                    other.data_type()
                )
            }
        }
    }
}
