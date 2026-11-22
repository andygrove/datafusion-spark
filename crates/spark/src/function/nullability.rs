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
//! Every function that uses [`SparkNullability`] is null-intolerant, so it declares only the
//! `nullable` rule, and [`spark_nullability!`] implements `simplify` from it:
//!
//! ```ignore
//! const NULLABILITY: SparkNullability = SparkNullability::null_intolerant();
//!
//! impl ScalarUDFImpl for SparkAscii {
//!     fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
//!         Ok(NULLABILITY.return_field(self.name(), DataType::Int32, args.arg_fields))
//!     }
//!
//!     spark_nullability!(NULLABILITY);
//! }
//! ```
//!
//! A function that computes its type in `return_type` uses
//! `spark_nullability!(NULLABILITY, return_type)`, which also implements
//! `return_field_from_args`.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion_expr::{Expr, ExprSchemable, ReturnFieldArgs, ScalarUDFImpl, lit};

/// The nullability of a null-intolerant function: when its result can be NULL, as Spark's
/// `Expression.nullable` declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SparkNullability {
    /// Nullable when any argument is: `children.exists(_.nullable)`, the default of
    /// `UnaryExpression`, `BinaryExpression` and `TernaryExpression`.
    Children,
    /// Always nullable, because the function can return NULL for non-NULL arguments.
    Always,
}

impl SparkNullability {
    /// A null-intolerant function whose result is nullable when an argument is. This is a
    /// `UnaryExpression`, `BinaryExpression` or `TernaryExpression` with
    /// `nullIntolerant = true` that does not override `nullable`.
    pub(crate) const fn null_intolerant() -> Self {
        Self::Children
    }

    /// A null-intolerant function that can return NULL for non-NULL arguments, so its result
    /// is always nullable. Spark overrides `nullable` to `true` for these.
    pub(crate) const fn null_intolerant_always_nullable() -> Self {
        Self::Always
    }

    /// Whether the result can be NULL, given the argument fields.
    pub(crate) fn nullable(&self, arg_fields: &[FieldRef]) -> bool {
        match self {
            Self::Children => arg_fields.iter().any(|f| f.is_nullable()),
            Self::Always => true,
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

    /// The return field of a function that computes its type in `ScalarUDFImpl::return_type`,
    /// for `ScalarUDFImpl::return_field_from_args`.
    pub(crate) fn return_field_from_type(
        &self,
        func: &dyn ScalarUDFImpl,
        args: &ReturnFieldArgs,
    ) -> Result<FieldRef> {
        let arg_types = args
            .arg_fields
            .iter()
            .map(|f| f.data_type().clone())
            .collect::<Vec<_>>();
        let data_type = func.return_type(&arg_types)?;
        Ok(self.return_field(func.name(), data_type, args.arg_fields))
    }

    /// For `ScalarUDFImpl::simplify`: replaces a call that has a NULL literal argument with a
    /// NULL literal of its return type, as Spark's `NullPropagation` does for a
    /// null-intolerant expression. Otherwise returns the arguments unchanged.
    ///
    /// This is only an optimization: `invoke_with_args` must return NULL for such a call too.
    pub(crate) fn simplify(
        &self,
        func: &dyn ScalarUDFImpl,
        args: Vec<Expr>,
        info: &SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        if !args.iter().any(is_null_literal) {
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
        match ScalarValue::try_new_null(return_field.data_type()) {
            Ok(null) => Ok(ExprSimplifyResult::Simplified(lit(null))),
            // A type that has no NULL `ScalarValue` is left to `invoke_with_args`
            Err(_) => Ok(ExprSimplifyResult::Original(args)),
        }
    }
}

/// Implements `ScalarUDFImpl` methods from a function's [`SparkNullability`], inside its
/// `impl ScalarUDFImpl` block:
///
/// - `spark_nullability!(NULLABILITY)` implements `simplify` with
///   [`SparkNullability::simplify`].
/// - `spark_nullability!(NULLABILITY, return_type)` also implements `return_field_from_args`
///   with [`SparkNullability::return_field_from_type`], for a function that computes its type
///   in `return_type`.
macro_rules! spark_nullability {
    ($nullability:expr) => {
        fn simplify(
            &self,
            args: Vec<datafusion_expr::Expr>,
            info: &datafusion_expr::simplify::SimplifyContext,
        ) -> datafusion_common::Result<datafusion_expr::simplify::ExprSimplifyResult> {
            $nullability.simplify(self, args, info)
        }
    };
    ($nullability:expr, return_type) => {
        $crate::function::nullability::spark_nullability!($nullability);

        fn return_field_from_args(
            &self,
            args: datafusion_expr::ReturnFieldArgs,
        ) -> datafusion_common::Result<arrow::datatypes::FieldRef> {
            $nullability.return_field_from_type(self, &args)
        }
    };
}
pub(crate) use spark_nullability;

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
    fn test_nullable_always() {
        let always = SparkNullability::null_intolerant_always_nullable();
        assert!(always.nullable(&[]));
        assert!(always.nullable(&fields(&[false])));
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

    /// The return field of `func` for arguments of `types`, where only the argument at
    /// `nullable_arg` (if any) is nullable.
    fn return_field_of(
        func: &dyn ScalarUDFImpl,
        types: &[DataType],
        nullable_arg: Option<usize>,
    ) -> FieldRef {
        let arg_fields = types
            .iter()
            .enumerate()
            .map(|(i, t)| {
                Arc::new(Field::new(
                    format!("a{i}"),
                    t.clone(),
                    nullable_arg == Some(i),
                ))
            })
            .collect::<Vec<_>>();
        let scalar_arguments = vec![None; types.len()];
        func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &arg_fields,
            scalar_arguments: &scalar_arguments,
        })
        .unwrap()
    }

    /// Functions that moved to [`SparkNullability`], with the nullability Spark declares.
    /// A function with `always_nullable` can return NULL for non-NULL arguments. Any other
    /// is nullable only when an argument is.
    #[test]
    fn test_return_field_nullability_of_null_intolerant_functions() {
        use crate::function::array::array_position::SparkArrayPosition;
        use crate::function::array::array_remove::SparkArrayRemove;
        use crate::function::bitwise::bit_count::SparkBitCount;
        use crate::function::datetime::dayofweek::SparkDayOfWeek;
        use crate::function::datetime::extract::{SparkHour, SparkMinute, SparkSecond};
        use crate::function::datetime::make_dt_interval::SparkMakeDtInterval;
        use crate::function::datetime::timestamp_micros::SparkTimestampMicros;
        use crate::function::datetime::weekday::SparkWeekDay;
        use crate::function::math::atan2::SparkAtan2;
        use crate::function::math::floor::SparkFloor;
        use crate::function::math::hex::SparkHex;
        use crate::function::math::hypot::SparkHypot;
        use crate::function::math::negative::SparkNegative;
        use crate::function::math::pow::SparkPow;
        use crate::function::math::trigonometry::{SparkCsc, SparkSec};
        use crate::function::string::contains::SparkContains;
        use crate::function::string::make_valid_utf8::SparkMakeValidUtf8;
        use crate::function::string::soundex::SparkSoundex;
        use crate::function::string::space::SparkSpace;

        let list = DataType::List(Arc::new(Field::new("item", DataType::Int64, false)));
        let timestamp =
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
        let int32 = DataType::Int32;
        let int64 = DataType::Int64;
        let float64 = DataType::Float64;
        let utf8 = DataType::Utf8;

        // (function, argument types, always nullable)
        let cases: Vec<(Box<dyn ScalarUDFImpl>, Vec<DataType>, bool)> = vec![
            // Nullable only when an argument is. Several of these were reported as
            // always nullable before they moved to `SparkNullability`
            (
                Box::new(SparkArrayPosition::new()),
                vec![list.clone(), int64.clone()],
                false,
            ),
            (
                Box::new(SparkArrayRemove::new()),
                vec![list, int64.clone()],
                false,
            ),
            (
                Box::new(SparkAtan2::new()),
                vec![float64.clone(), float64.clone()],
                false,
            ),
            (
                Box::new(SparkContains::new()),
                vec![utf8.clone(), utf8.clone()],
                false,
            ),
            (Box::new(SparkHex::new()), vec![int64.clone()], false),
            (Box::new(SparkHour::new()), vec![timestamp.clone()], false),
            (Box::new(SparkMinute::new()), vec![timestamp.clone()], false),
            (Box::new(SparkSecond::new()), vec![timestamp], false),
            (
                Box::new(SparkHypot::new()),
                vec![float64.clone(), float64.clone()],
                false,
            ),
            (Box::new(SparkNegative::new()), vec![int32.clone()], false),
            (
                Box::new(SparkPow::new()),
                vec![float64.clone(), float64.clone()],
                false,
            ),
            (Box::new(SparkSoundex::new()), vec![utf8.clone()], false),
            (Box::new(SparkSpace::new()), vec![int32.clone()], false),
            (Box::new(SparkBitCount::new()), vec![int32.clone()], false),
            (
                Box::new(SparkDayOfWeek::new()),
                vec![DataType::Date32],
                false,
            ),
            (Box::new(SparkWeekDay::new()), vec![DataType::Date32], false),
            (
                Box::new(SparkTimestampMicros::new()),
                vec![int64.clone()],
                false,
            ),
            // Nullable even for non-nullable arguments: `floor` and `csc` are
            // `UnaryMathExpression`s, and an invalid value gives NULL in the others
            (Box::new(SparkFloor::new()), vec![float64.clone()], true),
            (Box::new(SparkCsc::new()), vec![float64.clone()], true),
            (Box::new(SparkSec::new()), vec![float64.clone()], true),
            (Box::new(SparkMakeValidUtf8::new()), vec![utf8], true),
            (
                Box::new(SparkMakeDtInterval::new()),
                vec![int32.clone(), int32.clone(), int32, float64],
                true,
            ),
        ];

        for (func, types, always_nullable) in &cases {
            let name = func.name();
            assert_eq!(
                return_field_of(func.as_ref(), types, None).is_nullable(),
                *always_nullable,
                "{name} with no nullable argument"
            );
            for nullable_arg in 0..types.len() {
                assert!(
                    return_field_of(func.as_ref(), types, Some(nullable_arg))
                        .is_nullable(),
                    "{name} with argument {nullable_arg} nullable"
                );
            }
        }
    }
}
