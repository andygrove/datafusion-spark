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

const DAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/// `day` is 0 for Monday through 6 for Sunday.
fn day_number_to_name(day: i32) -> Option<&'static str> {
    DAY_NAMES.get(day as usize).copied()
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
            signature: date_signature(),
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
        date_part_name(arg, DatePart::DayOfWeekMonday0, day_number_to_name)
    }
}
