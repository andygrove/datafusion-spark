// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{AsArray, StringArray};
use arrow::compute::{DatePart, date_part};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::{NativeType, logical_date};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, TypeSignatureClass, Volatility,
};

const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// `day` is 0 for Monday through 6 for Sunday.
fn day_number_to_name(day: i32) -> Option<&'static str> {
    usize::try_from(day)
        .ok()
        .and_then(|d| DAY_NAMES.get(d))
        .copied()
}

/// Spark-compatible `dayname` expression (Spark 4.0+).
/// Returns the three-letter abbreviated English day name (`Mon` to `Sun`) of a date.
/// Timestamps are cast to dates first, like Spark's implicit cast.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#dayname>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDayName {
    signature: Signature,
}

impl Default for SparkDayName {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDayName {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![Coercion::new_implicit(
                    TypeSignatureClass::Native(logical_date()),
                    vec![TypeSignatureClass::Timestamp],
                    NativeType::Date,
                )],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for SparkDayName {
    fn name(&self) -> &str {
        "dayname"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
        Ok(Arc::new(Field::new(self.name(), DataType::Utf8, nullable)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        match arg {
            ColumnarValue::Scalar(scalar) => {
                if scalar.is_null() {
                    return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)));
                }
                let arr = scalar.to_array_of_size(1)?;
                let day_arr = date_part(&arr, DatePart::DayOfWeekMonday0)?;
                let day_val = day_arr
                    .as_primitive::<arrow::datatypes::Int32Type>()
                    .value(0);
                let name = day_number_to_name(day_val).map(|s| s.to_string());
                Ok(ColumnarValue::Scalar(ScalarValue::Utf8(name)))
            }
            ColumnarValue::Array(arr) => {
                let day_arr = date_part(&arr, DatePart::DayOfWeekMonday0)?;
                let int_arr = day_arr.as_primitive::<arrow::datatypes::Int32Type>();

                let result: StringArray = int_arr
                    .iter()
                    .map(|maybe_day| maybe_day.and_then(day_number_to_name))
                    .collect();

                Ok(ColumnarValue::Array(Arc::new(result)))
            }
        }
    }
}
