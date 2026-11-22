// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, FieldRef};
use datafusion_common::types::{NativeType, logical_date};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

use super::dayofweek::{epoch_day_field, week_day};
use crate::function::nullability::{SparkNullability, spark_nullability};

/// Spark's `WeekDay` is a null-intolerant `UnaryExpression`.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

/// Spark-compatible `weekday` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#weekday>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Returns the day of the week as an INT from 0 (Monday) to 6 (Sunday). \
        This differs from `dayofweek`, which counts from 1 (Sunday) to 7 (Saturday). \
        DataFusion has no built-in `weekday`. Timestamps are cast to DATE first. \
        Strings are not implicitly cast as they are in Spark ([#41](https://github.com/apache/datafusion-spark/issues/41)). Returns NULL if the input is NULL.",
    syntax_example = "weekday(date)",
    sql_example = r#"```sql
> SELECT weekday(DATE '2009-07-30');
3
> SELECT weekday(TIMESTAMP '2024-01-07 23:59:59');
6
> SELECT weekday(NULL::DATE);
NULL
```"#,
    argument(
        name = "date",
        description = "DATE or TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkWeekDay {
    signature: Signature,
}

impl Default for SparkWeekDay {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkWeekDay {
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

impl ScalarUDFImpl for SparkWeekDay {
    fn name(&self) -> &str {
        "weekday"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(NULLABILITY.return_field(self.name(), DataType::Int32, args.arg_fields))
    }

    spark_nullability!(NULLABILITY);

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [arg] = take_function_args(self.name(), args.args)?;
        epoch_day_field(self.name(), arg, week_day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Date32Array, Int32Array};
    use arrow::datatypes::{Field, Int32Type};
    use datafusion_common::ScalarValue;
    use std::sync::Arc;

    #[test]
    fn test_weekday_return_field_nullability_matches_input() {
        let func = SparkWeekDay::new();

        let non_nullable_arg = Arc::new(Field::new("arg", DataType::Date32, false));
        let nullable_arg = Arc::new(Field::new("arg", DataType::Date32, true));

        let non_nullable_out = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[Arc::clone(&non_nullable_arg)],
                scalar_arguments: &[None],
            })
            .expect("non-nullable arg should succeed");
        assert_eq!(non_nullable_out.data_type(), &DataType::Int32);
        assert!(!non_nullable_out.is_nullable());

        let nullable_out = func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[Arc::clone(&nullable_arg)],
                scalar_arguments: &[None],
            })
            .expect("nullable arg should succeed");
        assert_eq!(nullable_out.data_type(), &DataType::Int32);
        assert!(nullable_out.is_nullable());
    }

    #[test]
    fn test_weekday_scalar() -> Result<()> {
        let func = SparkWeekDay::new();

        // 2024-03-15 is a Friday -> Spark weekday = 4 (Mon=0).
        let result = func.invoke_with_args(ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(ScalarValue::Date32(Some(19797)))],
            arg_fields: vec![Arc::new(Field::new("arg", DataType::Date32, true))],
            number_rows: 1,
            return_field: Arc::new(Field::new("weekday", DataType::Int32, true)),
            config_options: Arc::new(Default::default()),
        })?;
        match result {
            ColumnarValue::Scalar(ScalarValue::Int32(Some(v))) => assert_eq!(v, 4),
            other => panic!("unexpected result: {other:?}"),
        }

        // NULL input -> NULL output.
        let result = func.invoke_with_args(ScalarFunctionArgs {
            args: vec![ColumnarValue::Scalar(ScalarValue::Date32(None))],
            arg_fields: vec![Arc::new(Field::new("arg", DataType::Date32, true))],
            number_rows: 1,
            return_field: Arc::new(Field::new("weekday", DataType::Int32, true)),
            config_options: Arc::new(Default::default()),
        })?;
        match result {
            ColumnarValue::Scalar(ScalarValue::Int32(None)) => {}
            other => panic!("unexpected result: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_weekday_array() -> Result<()> {
        let func = SparkWeekDay::new();

        // 2024-01-01 Mon(0), 2024-01-06 Sat(5), 2024-01-07 Sun(6), NULL.
        let input = Date32Array::from(vec![Some(19723), Some(19728), Some(19729), None]);
        let result = func.invoke_with_args(ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(input))],
            arg_fields: vec![Arc::new(Field::new("arg", DataType::Date32, true))],
            number_rows: 4,
            return_field: Arc::new(Field::new("weekday", DataType::Int32, true)),
            config_options: Arc::new(Default::default()),
        })?;
        match result {
            ColumnarValue::Array(arr) => {
                let expected = Int32Array::from(vec![Some(0), Some(5), Some(6), None]);
                assert_eq!(arr.as_primitive::<Int32Type>(), &expected);
            }
            other => panic!("unexpected result: {other:?}"),
        }

        Ok(())
    }
}
