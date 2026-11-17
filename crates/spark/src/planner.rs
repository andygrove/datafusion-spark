// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::Result;
use datafusion_expr::Expr;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::planner::{ExprPlanner, PlannerResult};
#[cfg(feature = "sql")]
use datafusion_expr::{
    ExprSchemable, planner::RawBinaryExpr, sqlparser::ast::BinaryOperator,
};

/// An [`ExprPlanner`] that plans SQL syntax using Apache Spark semantics.
///
/// Currently this maps:
///
/// * `EXTRACT(field FROM expr)` to the Spark `date_part` function
/// * `SUBSTRING(expr FROM start FOR len)` to the Spark `substring` function
///
/// It is registered automatically by
/// `SessionStateBuilderSpark::with_spark_features` (requires the `core`
/// feature), or you can register it yourself with
/// `SessionContext::register_expr_planner`.
#[derive(Default, Debug)]
pub struct SparkFunctionPlanner;

impl ExprPlanner for SparkFunctionPlanner {
    #[cfg(feature = "sql")]
    fn plan_binary_op(
        &self,
        expr: RawBinaryExpr,
        schema: &datafusion_common::DFSchema,
    ) -> Result<PlannerResult<RawBinaryExpr>> {
        if expr.op != BinaryOperator::Divide {
            return Ok(PlannerResult::Original(expr));
        }

        let left_type = expr.left.get_type(schema)?;
        let right_type = expr.right.get_type(schema)?;

        // Spark keeps DECIMAL arithmetic for DECIMAL / DECIMAL and for DECIMAL
        // divided by an integral type, and coerces both operands to DOUBLE as
        // soon as a floating point type is involved:
        //
        //   CAST(1 AS DECIMAL(10,2)) / 2                 -> DECIMAL
        //   CAST(1 AS DECIMAL(10,2)) / 2.0               -> DECIMAL
        //   CAST(1 AS DECIMAL(10,2)) / CAST(2 AS DOUBLE) -> DOUBLE
        //
        // DataFusion's `/` derives the same precision and scale for the DECIMAL
        // cases, but it does not agree with Spark everywhere (for example on a
        // DECIMAL division by zero in non-ANSI mode, where Spark returns NULL);
        // those differences are tracked separately.
        //
        // Operands that do not look numeric here also keep the `/` operator. That
        // includes a `NULL` literal and string operands (`1 / NULL`, `'4' / 2`),
        // which Spark casts to DOUBLE; handling those needs a wider coercion
        // change than this planner can make.
        let keeps_decimal_division = (left_type.is_decimal() && right_type.is_decimal())
            || (left_type.is_decimal() && right_type.is_integer())
            || (left_type.is_integer() && right_type.is_decimal());

        if !left_type.is_numeric() || !right_type.is_numeric() || keeps_decimal_division {
            return Ok(PlannerResult::Original(expr));
        }

        Ok(PlannerResult::Planned(Expr::ScalarFunction(
            ScalarFunction::new_udf(
                crate::function::math::spark_divide(),
                vec![expr.left, expr.right],
            ),
        )))
    }

    fn plan_extract(&self, args: Vec<Expr>) -> Result<PlannerResult<Vec<Expr>>> {
        Ok(PlannerResult::Planned(Expr::ScalarFunction(
            ScalarFunction::new_udf(crate::function::datetime::date_part(), args),
        )))
    }

    fn plan_substring(&self, args: Vec<Expr>) -> Result<PlannerResult<Vec<Expr>>> {
        Ok(PlannerResult::Planned(Expr::ScalarFunction(
            ScalarFunction::new_udf(crate::function::string::substring(), args),
        )))
    }
}
