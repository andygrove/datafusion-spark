// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion_common::types::{NativeType, logical_int32};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `date_from_unix_date` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_from_unix_date>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the date that is `days` days after 1970-01-01. This is the inverse of `unix_date`. \
        DataFusion has no built-in equivalent. Negative values give dates before the epoch. \
        Any INT is a valid date, so the function never errors. Returns NULL if the input is NULL.",
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
        description = "Integer number of days since 1970-01-01. Other integer types are cast to INT. NULL returns NULL."
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
        let int32 = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int32()),
            vec![TypeSignatureClass::Integer],
            NativeType::Int32,
        );
        Self {
            signature: Signature::coercible(vec![int32], Volatility::Immutable),
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
        Ok(DataType::Date32)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [days] = take_function_args(self.name(), args.args)?;
        match days {
            // Date32 is Int32 days since the epoch, so this cast reuses the buffer.
            ColumnarValue::Array(array) => {
                Ok(ColumnarValue::Array(cast(&array, &DataType::Date32)?))
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
