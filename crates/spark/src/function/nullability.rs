// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! A function's nullability, declared as Spark declares it.
//!
//! Spark describes it with two members of `Expression`:
//!
//! - `nullable`: whether the result can be NULL. `UnaryExpression`, `BinaryExpression` and
//!   `TernaryExpression` take it from their children (`children.exists(_.nullable)`), and an
//!   expression that can return NULL for non-NULL inputs overrides it, as `next_day` does.
//! - `nullIntolerant` (the `NullIntolerant` trait before Spark 4.0): a NULL input always gives a
//!   NULL output. The `NullPropagation` optimizer rule uses it to replace a call that has a NULL
//!   literal argument with a NULL literal.
//!
//! These are independent: a null-intolerant expression can still be nullable for non-NULL
//! inputs.
//!
//! A function here declares both in a [`SparkNullability`] and uses it from `ScalarUDFImpl`:
//!
//! ```ignore
//! const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();
//!
//! fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
//!     Ok(NULLABILITY.return_field(self.name(), DataType::Int32, args.arg_fields))
//! }
//!
//! fn simplify(&self, args: Vec<Expr>, info: &SimplifyContext) -> Result<ExprSimplifyResult> {
//!     NULLABILITY.simplify(self, args, info)
//! }
//! ```

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{Expr, ExprSchemable, ReturnFieldArgs, ScalarUDFImpl, lit};

/// When a function's result can be NULL: Spark's `Expression.nullable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(
    dead_code,
    reason = "declared for the functions that move onto SparkNullability next (#383)"
)]
pub(crate) enum SparkNullable {
    /// When any argument can be NULL: `children.exists(_.nullable)`, the default of
    /// `UnaryExpression`, `BinaryExpression` and `TernaryExpression`.
    Children,
    /// Always, because the function can return NULL for non-NULL arguments.
    Always,
    /// Never.
    Never,
}

/// A function's nullability: Spark's `nullable` and `nullIntolerant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SparkNullability {
    nullable: SparkNullable,
    null_intolerant: bool,
}

impl SparkNullability {
    /// A null-intolerant function whose result is nullable when an argument is. This is a
    /// `UnaryExpression`, `BinaryExpression` or `TernaryExpression` with
    /// `nullIntolerant = true` that does not override `nullable`.
    pub(crate) const fn null_intolerant() -> Self {
        Self::new(SparkNullable::Children, true)
    }

    /// Declares `nullable` and `nullIntolerant` separately.
    pub(crate) const fn new(nullable: SparkNullable, null_intolerant: bool) -> Self {
        Self {
            nullable,
            null_intolerant,
        }
    }

    /// Whether the result can be NULL, given the argument fields.
    pub(crate) fn nullable(&self, arg_fields: &[FieldRef]) -> bool {
        match self.nullable {
            SparkNullable::Children => arg_fields.iter().any(|f| f.is_nullable()),
            SparkNullable::Always => true,
            SparkNullable::Never => false,
        }
    }

    /// The return field, for `ScalarUDFImpl::return_field_from_args`.
    pub(crate) fn return_field(
        &self,
        name: &str,
        data_type: DataType,
        arg_fields: &[FieldRef],
    ) -> FieldRef {
        Arc::new(Field::new(name, data_type, self.nullable(arg_fields)))
    }

    /// For `ScalarUDFImpl::simplify`: replaces a call to a null-intolerant function that has
    /// a NULL literal argument with a NULL literal of its return type, as Spark's
    /// `NullPropagation` does. Otherwise returns the arguments unchanged.
    ///
    /// This is only an optimization: `invoke_with_args` must return NULL for such a call too.
    pub(crate) fn simplify<F: ScalarUDFImpl + ?Sized>(
        &self,
        func: &F,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        if !self.null_intolerant || !args.iter().any(is_null_literal) {
            return Ok(ExprSimplifyResult::Original(args));
        }
        let arg_fields = args
            .iter()
            .map(|arg| Ok(arg.to_field(info.schema())?.1))
            .collect::<Result<Vec<_>>>()?;
        let scalar_arguments = args
            .iter()
            .map(|arg| match arg {
                Expr::Literal(value, _) => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        let return_field = func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })?;
        let null = ScalarValue::try_from(return_field.data_type())?;
        Ok(ExprSimplifyResult::Simplified(lit(null)))
    }
}

fn is_null_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(value, _) if value.is_null())
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_common::{DFSchema, ScalarValue};
    use datafusion_expr::col;

    use crate::function::string::ascii::SparkAscii;

    fn fields(nullable: &[bool]) -> Vec<FieldRef> {
        nullable
            .iter()
            .enumerate()
            .map(|(i, n)| Arc::new(Field::new(format!("a{i}"), DataType::Utf8, *n)))
            .collect()
    }

    #[test]
    fn test_nullable_from_children() {
        let nullability = SparkNullability::null_intolerant();
        assert!(!nullability.nullable(&[]));
        assert!(!nullability.nullable(&fields(&[false, false])));
        assert!(nullability.nullable(&fields(&[false, true])));
    }

    #[test]
    fn test_nullable_always_and_never() {
        let always = SparkNullability::new(SparkNullable::Always, true);
        assert!(always.nullable(&fields(&[false])));
        let never = SparkNullability::new(SparkNullable::Never, false);
        assert!(!never.nullable(&fields(&[true])));
    }

    #[test]
    fn test_return_field() {
        let field = SparkNullability::null_intolerant().return_field(
            "f",
            DataType::Int32,
            &fields(&[true]),
        );
        assert_eq!(field.as_ref(), &Field::new("f", DataType::Int32, true));
    }

    fn schema() -> SimplifyContext {
        let schema = DFSchema::try_from(arrow::datatypes::Schema::new(vec![Field::new(
            "s",
            DataType::Utf8,
            true,
        )]))
        .unwrap();
        SimplifyContext::builder()
            .with_schema(Arc::new(schema))
            .build()
    }

    #[test]
    fn test_simplify_null_literal_to_typed_null() {
        let result = SparkNullability::null_intolerant()
            .simplify(
                &SparkAscii::new(),
                vec![lit(ScalarValue::Utf8(None))],
                &schema(),
            )
            .unwrap();
        let ExprSimplifyResult::Simplified(expr) = result else {
            panic!("expected a NULL literal");
        };
        assert_eq!(expr, lit(ScalarValue::Int32(None)));
    }

    #[test]
    fn test_simplify_keeps_non_null_arguments() {
        for args in [vec![col("s")], vec![lit("a")]] {
            let result = SparkNullability::null_intolerant()
                .simplify(&SparkAscii::new(), args.clone(), &schema())
                .unwrap();
            let ExprSimplifyResult::Original(original) = result else {
                panic!("expected the arguments unchanged");
            };
            assert_eq!(original, args);
        }
    }

    #[test]
    fn test_simplify_not_null_intolerant() {
        let args = vec![lit(ScalarValue::Utf8(None))];
        let result = SparkNullability::new(SparkNullable::Children, false)
            .simplify(&SparkAscii::new(), args.clone(), &schema())
            .unwrap();
        assert!(matches!(result, ExprSimplifyResult::Original(a) if a == args));
    }
}
