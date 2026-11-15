// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::compute;
use arrow::datatypes::{DataType, Date32Type, Field, FieldRef};
use datafusion_common::cast::{
    as_date32_array, as_int8_array, as_int16_array, as_int32_array,
};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, TypeSignature, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Spark-compatible `date_add` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_add>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the date that is `num_days` days after `start_date`. \
        A negative `num_days` moves the date backwards. DataFusion has no built-in `date_add`. \
        Returns NULL if either input is NULL. \
        The arithmetic wraps around on overflow instead of returning an error. \
        `num_days` must be INT, SMALLINT or TINYINT. Integer literals are BIGINT in DataFusion, \
        so cast them (for example `1::INT`; see [#97](https://github.com/apache/datafusion-spark/issues/97)). `start_date` must be a DATE; strings are not implicitly cast as they are in Spark ([#41](https://github.com/apache/datafusion-spark/issues/41)).",
    syntax_example = "date_add(start_date, num_days)",
    alternative_syntax = "dateadd(start_date, num_days)",
    sql_example = r#"```sql
> SELECT date_add(DATE '2016-07-30', 1::INT);
2016-07-31
> SELECT date_add(DATE '2016-07-30', -5::INT);
2016-07-25
> SELECT date_add(DATE '2016-07-30', NULL::INT);
NULL
```"#,
    argument(
        name = "start_date",
        description = "DATE expression. NULL returns NULL."
    ),
    argument(
        name = "num_days",
        description = "INT, SMALLINT or TINYINT number of days to add. Negative values subtract days. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDateAdd {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkDateAdd {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDateAdd {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![DataType::Date32, DataType::Int8]),
                    TypeSignature::Exact(vec![DataType::Date32, DataType::Int16]),
                    TypeSignature::Exact(vec![DataType::Date32, DataType::Int32]),
                ],
                Volatility::Immutable,
            ),
            aliases: vec!["dateadd".to_string()],
        }
    }
}

impl ScalarUDFImpl for SparkDateAdd {
    fn name(&self) -> &str {
        "date_add"
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("Use return_field_from_args in this case instead.")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
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
        make_scalar_function(spark_date_add, vec![])(&args.args)
    }
}

fn spark_date_add(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [date_arg, days_arg] = take_function_args("date_add", args)?;
    let date_array = as_date32_array(date_arg)?;
    let result = match days_arg.data_type() {
        DataType::Int8 => {
            let days_array = as_int8_array(days_arg)?;
            compute::binary::<_, _, _, Date32Type>(
                date_array,
                days_array,
                |date, days| date.wrapping_add(days as i32),
            )?
        }
        DataType::Int16 => {
            let days_array = as_int16_array(days_arg)?;
            compute::binary::<_, _, _, Date32Type>(
                date_array,
                days_array,
                |date, days| date.wrapping_add(days as i32),
            )?
        }
        DataType::Int32 => {
            let days_array = as_int32_array(days_arg)?;
            compute::binary::<_, _, _, Date32Type>(
                date_array,
                days_array,
                |date, days| date.wrapping_add(days),
            )?
        }
        _ => {
            return internal_err!(
                "Spark `date_add` function: argument must be int8, int16, int32, got {:?}",
                days_arg.data_type()
            );
        }
    };
    Ok(Arc::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_add_non_nullable_inputs() {
        let func = SparkDateAdd::new();
        let args = &[
            Arc::new(Field::new("date", DataType::Date32, false)),
            Arc::new(Field::new("num", DataType::Int8, false)),
        ];

        let ret_field = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: args,
                scalar_arguments: &[None, None],
            })
            .unwrap();

        assert_eq!(ret_field.data_type(), &DataType::Date32);
        assert!(!ret_field.is_nullable());
    }

    #[test]
    fn test_date_add_nullable_inputs() {
        let func = SparkDateAdd::new();
        let args = &[
            Arc::new(Field::new("date", DataType::Date32, false)),
            Arc::new(Field::new("num", DataType::Int16, true)),
        ];

        let ret_field = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: args,
                scalar_arguments: &[None, None],
            })
            .unwrap();

        assert_eq!(ret_field.data_type(), &DataType::Date32);
        assert!(ret_field.is_nullable());
    }
}
