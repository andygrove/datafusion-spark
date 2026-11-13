// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::compute::DatePart;
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
};

use super::date_part_name::{date_part_name, date_signature};

const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn month_number_to_name(month: i32) -> Option<&'static str> {
    MONTH_NAMES.get((month - 1) as usize).copied()
}

/// Spark-compatible `monthname` expression.
/// Returns the three-letter abbreviated month name from a date or timestamp.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#monthname>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkMonthName {
    signature: Signature,
}

impl Default for SparkMonthName {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkMonthName {
    pub fn new() -> Self {
        Self {
            signature: date_signature(),
        }
    }
}

impl ScalarUDFImpl for SparkMonthName {
    fn name(&self) -> &str {
        "monthname"
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
        date_part_name(arg, DatePart::Month, month_number_to_name)
    }
}
