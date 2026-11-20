// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, FieldRef};
use datafusion_common::types::{NativeType, logical_string};
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignatureClass, Volatility,
};
use datafusion_functions::string::ascii::ascii;
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

use crate::function::nullability::{SparkNullability, spark_nullability};

/// Spark's `Ascii` is a null-intolerant `UnaryExpression`.
const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();

/// Spark-compatible `ascii` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#ascii>
#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the numeric code point of the first character of `str`. \
        Despite the name, this is the full Unicode code point, not just ASCII (`ascii('😀')` is 128512). \
        An empty string returns 0, and NULL returns NULL. \
        Unlike DataFusion's `ascii`, numeric input is accepted and cast to a string first, \
        so `ascii(2)` is the code point of `'2'`.",
    syntax_example = "ascii(str)",
    sql_example = r#"```sql
> SELECT ascii('Spark');
83
> SELECT ascii('');
0
> SELECT ascii(2);
50
```"#,
    argument(
        name = "str",
        description = "String expression. Numeric values are cast to a string. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkAscii {
    signature: Signature,
}

impl Default for SparkAscii {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkAscii {
    pub fn new() -> Self {
        // Spark's ascii uses ImplicitCastInputTypes with StringType,
        // which allows numeric types to be implicitly cast to String.
        // See: https://github.com/apache/spark/blob/master/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/stringExpressions.scala
        let string_coercion = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_string()),
            vec![TypeSignatureClass::Numeric],
            NativeType::String,
        );

        Self {
            signature: Signature::coercible(vec![string_coercion], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkAscii {
    fn name(&self) -> &str {
        "ascii"
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
        make_scalar_function(ascii, vec![])(&args.args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use std::sync::Arc;

    #[test]
    fn test_return_field_nullable_input() {
        let ascii_func = SparkAscii::new();
        let nullable_field = Arc::new(Field::new("input", DataType::Utf8, true));

        let result = ascii_func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[nullable_field],
                scalar_arguments: &[],
            })
            .unwrap();

        assert_eq!(result.data_type(), &DataType::Int32);
        assert!(
            result.is_nullable(),
            "Output should be nullable when input is nullable"
        );
    }

    #[test]
    fn test_return_field_non_nullable_input() {
        let ascii_func = SparkAscii::new();
        let non_nullable_field = Arc::new(Field::new("input", DataType::Utf8, false));

        let result = ascii_func
            .return_field_from_args(ReturnFieldArgs {
                arg_fields: &[non_nullable_field],
                scalar_arguments: &[],
            })
            .unwrap();

        assert_eq!(result.data_type(), &DataType::Int32);
        assert!(
            !result.is_nullable(),
            "Output should not be nullable when input is not nullable"
        );
    }
}
