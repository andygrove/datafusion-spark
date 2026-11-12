// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion_common::types::{NativeType, logical_int32};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignatureClass, Volatility,
};

/// Spark-compatible `date_from_unix_date(days)`.
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_from_unix_date>
///
/// Creates a date from the number of days since 1970-01-01. The argument is an
/// INT (other integer types are cast to it), and NULL gives NULL. Any INT is a
/// valid date, so there is no error case. This is the inverse of `unix_date`.
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Date32Array, Int32Array};

    use super::*;

    #[test]
    fn array_input_reinterprets_days() -> Result<()> {
        let input: arrow::array::ArrayRef =
            Arc::new(Int32Array::from(vec![Some(0), Some(1000), None, Some(-1)]));
        let out = cast(&input, &DataType::Date32)?;
        let expected = Date32Array::from(vec![Some(0), Some(1000), None, Some(-1)]);
        assert_eq!(out.to_data(), expected.to_data());
        Ok(())
    }
}
