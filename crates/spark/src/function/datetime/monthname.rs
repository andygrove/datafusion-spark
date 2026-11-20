// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::compute::DatePart;
use arrow::datatypes::{DataType, FieldRef};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature,
};
use datafusion_macros::user_doc;

use super::date_part_name::{date_part_name, date_signature};
use crate::function::nullability::{SparkNullability, spark_nullability};

const MONTH_NAMES: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

fn month_number_to_name(month: i32) -> Option<&'static str> {
    MONTH_NAMES.get((month - 1) as usize).copied()
}

/// Spark's `MonthName` is a null-intolerant `UnaryExpression`.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

/// Spark-compatible `monthname` expression (Spark 4.0+).
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#monthname>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the three-letter abbreviated English month name (`Jan` to `Dec`) of a date. \
        DataFusion has no built-in `monthname`. \
        Strings are implicitly cast to DATE. A malformed string is an error, even with ANSI mode off \
        where Spark returns NULL. \
        Timestamps are cast to DATE first, so the time of day is ignored. A timestamp with a time zone \
        is converted in its own time zone, not the session time zone that Spark uses. \
        Returns NULL if the input is NULL.",
    syntax_example = "monthname(date)",
    sql_example = r#"```sql
> SELECT monthname('2008-02-20'::DATE);
Feb
> SELECT monthname('2008-02-20');
Feb
> SELECT monthname('2024-03-15 12:34:56'::TIMESTAMP);
Mar
> SELECT monthname(NULL::DATE);
NULL
```"#,
    argument(
        name = "date",
        description = "DATE, TIMESTAMP or string expression. NULL returns NULL."
    )
)]
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
        Ok(NULLABILITY.return_field(self.name(), DataType::Utf8, args.arg_fields))
    }

    spark_nullability!(NULLABILITY);

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        date_part_name(arg, DatePart::Month, month_number_to_name)
    }
}
