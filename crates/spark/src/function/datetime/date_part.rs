// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::types::logical_date;
use datafusion_common::{
    Result, ScalarValue, internal_err, plan_err, types::logical_string,
    utils::take_function_args,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Field names Spark accepts, one group per field, from `DatePart.parseExtractField`.
const SPARK_FIELDS: &[&str] = &[
    "year y years yr yrs",
    "yearofweek",
    "quarter qtr",
    "month mon mons months",
    "week w weeks",
    "day d days",
    "dayofweek dow",
    "dayofweek_iso dow_iso",
    "doy",
    "hour h hours hr hrs",
    "minute m min mins minutes",
    "second s sec seconds secs",
];

/// Spark-compatible `date_part` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#date_part>
#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = "Extracts a field from a date or timestamp, as an INT. \
        Unlike DataFusion's `date_part`, it accepts Spark's field names and abbreviations (case-insensitive), \
        and `DAYOFWEEK`/`DOW` counts from 1 (Sunday) to 7 (Saturday) where DataFusion's `dow` counts from 0 (Sunday). \
        `DAYOFWEEK_ISO`/`DOW_ISO` counts from 1 (Monday) to 7 (Sunday), and `YEAROFWEEK` is the ISO week-numbering year. \
        A field name Spark does not accept (such as `QUARTERS` or `DAYOFYEAR`) is an error at planning time. \
        A NULL field or source returns NULL. \
        `SECOND` returns whole seconds; Spark returns DECIMAL(8, 6) including the fraction of a second.",
    syntax_example = "date_part(field, source)",
    alternative_syntax = "datepart(field, source)",
    sql_example = r#"```sql
> SELECT date_part('YEAR', DATE '2000-01-01');
2000
> SELECT date_part('DOW', DATE '2000-01-01');
7
> SELECT date_part('MINS', TIMESTAMP '2000-01-01 12:30:45');
30
> SELECT date_part('year', NULL::TIMESTAMP);
NULL
```"#,
    argument(
        name = "field",
        description = "String literal naming the field to extract: `YEAR` (`Y`, `YEARS`, `YR`, `YRS`), `YEAROFWEEK`, \
            `QUARTER` (`QTR`), `MONTH` (`MON`, `MONS`, `MONTHS`), `WEEK` (`W`, `WEEKS`), `DAY` (`D`, `DAYS`), \
            `DAYOFWEEK` (`DOW`), `DAYOFWEEK_ISO` (`DOW_ISO`), `DOY`, `HOUR` (`H`, `HOURS`, `HR`, `HRS`), \
            `MINUTE` (`M`, `MIN`, `MINS`, `MINUTES`) or `SECOND` (`S`, `SEC`, `SECONDS`, `SECS`). NULL returns NULL."
    ),
    argument(
        name = "source",
        description = "DATE or TIMESTAMP expression. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkDatePart {
    signature: Signature,
    aliases: Vec<String>,
}

impl Default for SparkDatePart {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkDatePart {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                        Coercion::new_exact(TypeSignatureClass::Timestamp),
                    ]),
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                        Coercion::new_exact(TypeSignatureClass::Native(logical_date())),
                    ]),
                ],
                Volatility::Immutable,
            ),
            aliases: vec![String::from("datepart")],
        }
    }
}

impl ScalarUDFImpl for SparkDatePart {
    fn name(&self) -> &str {
        "date_part"
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

    // Nullable, like the integer parts of `date_part`, which this simplifies to,
    // because they can be NULL for a non-null input
    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("spark date_part should have been simplified to standard date_part")
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        let [part_expr, date_expr] = take_function_args(self.name(), args)?;

        let part = match part_expr.as_literal() {
            Some(ScalarValue::Utf8(Some(v)))
            | Some(ScalarValue::Utf8View(Some(v)))
            | Some(ScalarValue::LargeUtf8(Some(v))) => {
                let part = v.to_lowercase();
                if !SPARK_FIELDS.iter().any(|g| g.split(' ').any(|f| f == part)) {
                    let data_type = info.get_data_type(&date_expr)?;
                    return plan_err!(
                        "Cannot extract `{v}` from {data_type}. SQLSTATE: 42601"
                    );
                }
                part
            }
            // A NULL field returns NULL, like Spark
            Some(value) if value.is_null() => {
                return Ok(ExprSimplifyResult::Simplified(Expr::Literal(
                    ScalarValue::Int32(None),
                    None,
                )));
            }
            _ => {
                return internal_err!(
                    "First argument of `DATE_PART` must be non-null scalar Utf8"
                );
            }
        };

        // Map Spark-specific date part aliases to DataFusion ones.
        let part = match part.as_str() {
            "yearofweek" => "isoyear",
            "dayofweek" | "dow" => "dow1",
            "dayofweek_iso" | "dow_iso" => "isodow",
            other => other,
        };

        let part_expr = Expr::Literal(ScalarValue::new_utf8(part), None);

        let date_part_expr = Expr::ScalarFunction(ScalarFunction::new_udf(
            datafusion_functions::datetime::date_part(),
            vec![part_expr, date_expr],
        ));

        Ok(ExprSimplifyResult::Simplified(date_part_expr))
    }
}
