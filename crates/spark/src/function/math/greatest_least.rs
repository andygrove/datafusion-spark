// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark-compatible `greatest` and `least`.
//!
//! Spark evaluates the arguments in order, skips NULLs, and replaces its running result only
//! with a value that is strictly greater (or smaller) in its SQL ordering, so the result is NULL
//! only when every argument is NULL and the first of equal arguments wins. The SQL ordering
//! (`SQLOrderingUtil.compareDoubles`) treats `-0.0` as equal to `0.0` and NaN as greater than
//! every other value, at any depth of arrays and structs, and orders a NULL inside an array or
//! struct before every other value.
//!
//! DataFusion's built-in `greatest` and `least` differ: they compare floats by IEEE 754 total
//! order (so `-0.0 < 0.0`, and a NaN with the sign bit set is smaller than `-Infinity`), let a
//! later argument win a tie, compare the constant arguments before the columns, accept a single
//! argument, order a NULL inside an array or struct after every other value in `least`, and
//! coerce their arguments with DataFusion's comparison coercion, which for example compares a
//! number with a string. Spark rejects that.

use crate::function::array::float_normalize::{contains_float, normalize_array};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Datum, Scalar, make_comparator, new_null_array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::cmp::{gt, lt};
use arrow::compute::kernels::zip::zip;
use arrow::compute::{SortOptions, cast};
use arrow::datatypes::{DataType, Field, FieldRef, Fields, TimeUnit};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{DataFusionError, Result, ScalarValue, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;
use std::sync::Arc;

/// Spark-compatible `greatest` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#greatest>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the greatest value of all arguments, skipping NULLs. The result is \
        NULL only if every argument is NULL. At least two arguments are required. The \
        arguments are cast to a common type as in Spark: numbers widen (an integer and a FLOAT \
        widen to DOUBLE with `datafusion.execution.enable_ansi_mode` on and to FLOAT with it \
        off), integers and decimals widen to a decimal that holds both, a decimal and a float \
        widen to DOUBLE, a date and a timestamp widen to the timestamp, and arrays and structs \
        widen element by element. Strings are never promoted: a string and a number, a string \
        and a date, or a boolean and a number are an error, as are maps, which cannot be \
        ordered. NaN is greater than every other value and `-0.0` equals `0.0`, also inside \
        arrays and structs, and of equal values the first argument wins, so \
        `greatest(-0.0, 0.0)` is `-0.0`. A NULL inside an array or struct is smaller than \
        every other value. DataFusion's built-in `greatest` orders floats by IEEE 754 total \
        order, lets a later argument win a tie, accepts a single argument, and compares \
        numbers with strings.",
    syntax_example = "greatest(expr1, expr2[, ..., expr_n])",
    sql_example = r#"```sql
> SELECT greatest(10, 9, 2, 4, 3);
10
> SELECT greatest(1, NULL, 3);
3
> SELECT greatest(1.0, 'NaN'::double);
NaN
```"#,
    argument(
        name = "expr1, expr2, ..., expr_n",
        description = "Expressions of an orderable type that share a common type. NULLs are \
            skipped."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkGreatest {
    inner: GreatestLeast,
}

impl Default for SparkGreatest {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkGreatest {
    pub fn new() -> Self {
        Self {
            inner: GreatestLeast::new(true, false),
        }
    }

    /// Creates a `greatest` that coerces its arguments as Spark does with
    /// `datafusion.execution.enable_ansi_mode` set to `enable_ansi_mode`.
    pub fn new_with_ansi_mode(enable_ansi_mode: bool) -> Self {
        Self {
            inner: GreatestLeast::new(true, enable_ansi_mode),
        }
    }
}

/// Spark-compatible `least` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#least>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Returns the least value of all arguments, skipping NULLs. The result is \
        NULL only if every argument is NULL. At least two arguments are required. The \
        arguments are cast to a common type as in `greatest`. NaN is greater than every other \
        value and `-0.0` equals `0.0`, also inside arrays and structs, and of equal values the \
        first argument wins, so `least(0.0, -0.0)` is `0.0`. A NULL inside an array or struct \
        is smaller than every other value, so `least(array(1, NULL), array(1, 0))` is \
        `[1, NULL]`. DataFusion's built-in `least` orders floats by IEEE 754 total order, lets a \
        later argument win a tie, orders a NULL inside an array or struct after every other \
        value, accepts a single argument, and compares numbers with strings.",
    syntax_example = "least(expr1, expr2[, ..., expr_n])",
    sql_example = r#"```sql
> SELECT least(10, 9, 2, 4, 3);
2
> SELECT least(NULL, 2, NULL);
2
> SELECT least(1.0, 'NaN'::double);
1.0
```"#,
    argument(
        name = "expr1, expr2, ..., expr_n",
        description = "Expressions of an orderable type that share a common type. NULLs are \
            skipped."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkLeast {
    inner: GreatestLeast,
}

impl Default for SparkLeast {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkLeast {
    pub fn new() -> Self {
        Self {
            inner: GreatestLeast::new(false, false),
        }
    }

    /// Creates a `least` that coerces its arguments as Spark does with
    /// `datafusion.execution.enable_ansi_mode` set to `enable_ansi_mode`.
    pub fn new_with_ansi_mode(enable_ansi_mode: bool) -> Self {
        Self {
            inner: GreatestLeast::new(false, enable_ansi_mode),
        }
    }
}

macro_rules! impl_greatest_least_udf {
    ($udf:ident) => {
        impl ScalarUDFImpl for $udf {
            fn name(&self) -> &str {
                self.inner.name()
            }

            fn signature(&self) -> &Signature {
                &self.inner.signature
            }

            fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
                self.inner.coerce_types(arg_types)
            }

            fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
                self.inner.check_arg_count(arg_types.len(), true)?;
                Ok(arg_types[0].clone())
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
                self.inner.check_arg_count(args.arg_fields.len(), true)?;
                // As in Spark, the result is nullable only if every argument is.
                let nullable = args.arg_fields.iter().all(|f| f.is_nullable());
                Ok(Arc::new(Field::new(
                    self.name(),
                    args.arg_fields[0].data_type().clone(),
                    nullable,
                )))
            }

            fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
                Some(ScalarUDF::from(Self::new_with_ansi_mode(
                    config.execution.enable_ansi_mode,
                )))
            }

            fn invoke_with_args(
                &self,
                args: ScalarFunctionArgs,
            ) -> Result<ColumnarValue> {
                self.inner.invoke(args)
            }

            fn documentation(&self) -> Option<&Documentation> {
                self.doc()
            }
        }
    };
}

impl_greatest_least_udf!(SparkGreatest);
impl_greatest_least_udf!(SparkLeast);

/// The implementation shared by [`SparkGreatest`] and [`SparkLeast`].
#[derive(Debug, PartialEq, Eq, Hash)]
struct GreatestLeast {
    signature: Signature,
    greatest: bool,
    /// Whether to coerce as Spark's `AnsiTypeCoercion` does, rather than `TypeCoercion`.
    ansi_mode: bool,
}

/// Spark orders a NULL inside an array or struct before every other value.
const SORT_OPTIONS: SortOptions = SortOptions {
    descending: false,
    nulls_first: true,
};

impl GreatestLeast {
    fn new(greatest: bool, ansi_mode: bool) -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            greatest,
            ansi_mode,
        }
    }

    fn name(&self) -> &'static str {
        if self.greatest { "greatest" } else { "least" }
    }

    fn check_arg_count(&self, count: usize, planning: bool) -> Result<()> {
        if count > 1 {
            return Ok(());
        }
        let message = format!(
            "[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `{}` requires > 1 parameters but the \
             actual number is {count}. Please, refer to \
             'https://spark.apache.org/docs/latest/sql-ref-functions.html' for a fix.",
            self.name()
        );
        Err(if planning {
            DataFusionError::Plan(message)
        } else {
            DataFusionError::Execution(message)
        })
    }

    /// Spark's `TypeCoercion` rule for `Greatest` and `Least`: every argument is cast to
    /// `findWiderTypeWithoutStringPromotion` of all the argument types, and the result must be
    /// orderable.
    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        self.check_arg_count(arg_types.len(), true)?;
        let common = arg_types
            .iter()
            .skip(1)
            .try_fold(arg_types[0].clone(), |common, data_type| {
                wider_type(&common, data_type, self.ansi_mode)
            });
        let Some(common) = common else {
            return plan_err!(
                "[DATATYPE_MISMATCH.DATA_DIFF_TYPES] Input to `{}` should all be the same \
                 type, but it's [{}]",
                self.name(),
                arg_types
                    .iter()
                    .map(|t| format!("\"{}\"", spark_sql_type(t)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        };
        if !is_orderable(&common) {
            return plan_err!(
                "[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `{}` does not support ordering \
                 on type \"{}\"",
                self.name(),
                spark_sql_type(&common)
            );
        }
        Ok(vec![common; arg_types.len()])
    }

    fn invoke(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        self.check_arg_count(args.args.len(), false)?;
        let data_type = args.return_field.data_type();
        let all_scalars = args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));
        let rows = if all_scalars { 1 } else { args.number_rows };

        if data_type.is_null() {
            return Ok(if all_scalars {
                ColumnarValue::Scalar(ScalarValue::Null)
            } else {
                ColumnarValue::Array(new_null_array(&DataType::Null, rows))
            });
        }

        // The planner casts every argument to the common type, but a field name or the
        // nullability of a nested field can still differ when the function is called directly,
        // and `zip` needs identical types.
        let to_array = |arg: &ColumnarValue, rows: usize| -> Result<ArrayRef> {
            let array = arg.to_array(rows)?;
            Ok(if array.data_type() == data_type {
                array
            } else {
                cast(&array, data_type)?
            })
        };
        // Floats are compared in a copy where `-0.0` is `0.0` and every NaN is the canonical
        // NaN, where Arrow's total order agrees with Spark's. The result keeps the original.
        let normalize = contains_float(data_type);
        let comparable = |array: &ArrayRef| -> Result<ArrayRef> {
            if normalize {
                normalize_array(array)
            } else {
                Ok(Arc::clone(array))
            }
        };

        let mut result = to_array(&args.args[0], rows)?;
        let mut result_cmp = comparable(&result)?;
        for arg in &args.args[1..] {
            let candidate = match arg {
                // A NULL never replaces the result.
                ColumnarValue::Scalar(scalar) if scalar.is_null() => continue,
                ColumnarValue::Scalar(_) => {
                    let value = to_array(arg, 1)?;
                    let cmp = comparable(&value)?;
                    Candidate {
                        value,
                        cmp,
                        scalar: true,
                    }
                }
                ColumnarValue::Array(_) => {
                    let value = to_array(arg, rows)?;
                    let cmp = comparable(&value)?;
                    Candidate {
                        value,
                        cmp,
                        scalar: false,
                    }
                }
            };
            let take = self.replaces(&candidate, &result, &result_cmp)?;
            let take = BooleanArray::new(take, None);
            let new_result = zip(&take, candidate.value_datum().as_ref(), &result)?;
            result_cmp = if normalize {
                zip(&take, candidate.cmp_datum().as_ref(), &result_cmp)?
            } else {
                Arc::clone(&new_result)
            };
            result = new_result;
        }

        if all_scalars {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }

    /// For each row, whether `candidate` replaces the current result: it is not NULL, and
    /// either the result is NULL or the candidate ranks strictly before it.
    fn replaces(
        &self,
        candidate: &Candidate,
        current: &ArrayRef,
        current_cmp: &ArrayRef,
    ) -> Result<BooleanBuffer> {
        let len = current.len();
        let current_nulls = current.logical_nulls();
        let candidate_nulls = if candidate.scalar {
            // A NULL constant is skipped before this is called.
            None
        } else {
            candidate.value.logical_nulls()
        };

        if !current.data_type().is_nested() {
            let candidate_cmp = candidate.cmp_datum();
            let ordered = if self.greatest {
                gt(candidate_cmp.as_ref(), current_cmp)?
            } else {
                lt(candidate_cmp.as_ref(), current_cmp)?
            };
            // Where either side is NULL the comparison's value is arbitrary, but such rows are
            // decided by the null masks below.
            let mut take = ordered.values().clone();
            if let Some(nulls) = &current_nulls {
                take = &take | &!nulls.inner();
            }
            if let Some(nulls) = &candidate_nulls {
                take = &take & nulls.inner();
            }
            return Ok(take);
        }

        let compare =
            make_comparator(candidate.cmp.as_ref(), current_cmp.as_ref(), SORT_OPTIONS)?;
        let scalar = candidate.scalar;
        Ok(BooleanBuffer::collect_bool(len, |row| {
            let candidate_row = if scalar { 0 } else { row };
            if candidate_nulls
                .as_ref()
                .is_some_and(|nulls| nulls.is_null(candidate_row))
            {
                return false;
            }
            if current_nulls
                .as_ref()
                .is_some_and(|nulls| nulls.is_null(row))
            {
                return true;
            }
            let ordering = compare(candidate_row, row);
            if self.greatest {
                ordering.is_gt()
            } else {
                ordering.is_lt()
            }
        }))
    }
}

/// An argument after the first, with its values as they are and as they are compared. A
/// constant is a one-row array.
struct Candidate {
    value: ArrayRef,
    cmp: ArrayRef,
    scalar: bool,
}

impl Candidate {
    fn datum(&self, array: &ArrayRef) -> Box<dyn Datum> {
        if self.scalar {
            Box::new(Scalar::new(Arc::clone(array)))
        } else {
            Box::new(Arc::clone(array))
        }
    }

    fn value_datum(&self) -> Box<dyn Datum> {
        self.datum(&self.value)
    }

    fn cmp_datum(&self) -> Box<dyn Datum> {
        self.datum(&self.cmp)
    }
}

/// Spark's `findWiderTypeWithoutStringPromotionForTwo`: the tightest common type, then a
/// decimal wide enough for both, then the same for the elements of arrays and the fields of
/// structs. Strings are never promoted. `ansi_mode` picks between `AnsiTypeCoercion` and
/// `TypeCoercion`, which differ only in that ANSI widens an integer and a FLOAT to DOUBLE.
fn wider_type(left: &DataType, right: &DataType, ansi_mode: bool) -> Option<DataType> {
    tightest_common_type(left, right, ansi_mode)
        .or_else(|| wider_decimal_type(left, right))
        .or_else(|| complex_type(left, right, ansi_mode))
}

/// Spark's `findTightestCommonType`, on the Arrow types that Spark's types map to. Arrays and
/// structs are handled by [`complex_type`] with [`wider_type`], which gives the same result.
fn tightest_common_type(
    left: &DataType,
    right: &DataType,
    ansi_mode: bool,
) -> Option<DataType> {
    use DataType::*;
    if left == right {
        return Some(left.clone());
    }
    if let (Some(l), Some(r)) = (numeric_precedence(left), numeric_precedence(right)) {
        let wider = if l >= r { left } else { right };
        // With ANSI mode on, an integer and a FLOAT widen to DOUBLE, which holds every INT
        // exactly.
        return Some(if ansi_mode && *wider == Float32 {
            Float64
        } else {
            wider.clone()
        });
    }
    match (left, right) {
        (Null, other) | (other, Null) => Some(other.clone()),
        // Spark has one string type and one binary type.
        (l, r) if is_string(l) && is_string(r) => Some(wider_physical_type(l, r)),
        (l, r) if is_binary(l) && is_binary(r) => Some(wider_physical_type(l, r)),
        // The decimal must hold every value of the integral type.
        (int, Decimal128(p, s)) | (Decimal128(p, s), int) => {
            let digits = integral_digits(int)?;
            (*p as i16 - *s as i16 >= digits).then_some(Decimal128(*p, *s))
        }
        (Date32, ts @ Timestamp(_, _)) | (ts @ Timestamp(_, _), Date32) => {
            Some(ts.clone())
        }
        // TIMESTAMP_NTZ and TIMESTAMP widen to TIMESTAMP.
        (Timestamp(lu, ltz), Timestamp(ru, rtz)) => Some(Timestamp(
            finer_unit(*lu, *ru),
            ltz.clone().or_else(|| rtz.clone()),
        )),
        _ => None,
    }
}

/// Spark's `findWiderTypeForDecimal`, with `DecimalType.boundedPreferIntegralDigits`, which
/// gives up fractional digits rather than integral ones when the precision exceeds 38.
fn wider_decimal_type(left: &DataType, right: &DataType) -> Option<DataType> {
    use DataType::*;
    let (p1, s1, p2, s2) = match (left, right) {
        (Decimal128(p1, s1), Decimal128(p2, s2)) => {
            (*p1 as i16, *s1 as i16, *p2 as i16, *s2 as i16)
        }
        (int, Decimal128(p, s)) | (Decimal128(p, s), int)
            if integral_digits(int).is_some() =>
        {
            (integral_digits(int)?, 0, *p as i16, *s as i16)
        }
        (Float32 | Float64, Decimal128(_, _)) | (Decimal128(_, _), Float32 | Float64) => {
            return Some(Float64);
        }
        _ => return None,
    };
    let scale = s1.max(s2);
    let range = (p1 - s1).max(p2 - s2);
    let precision = scale + range;
    const MAX_PRECISION: i16 = 38;
    if precision <= MAX_PRECISION {
        Some(Decimal128(precision as u8, scale as i8))
    } else {
        let scale = (scale - (precision - MAX_PRECISION)).max(0);
        Some(Decimal128(MAX_PRECISION as u8, scale as i8))
    }
}

/// Spark's `findTypeForComplex` with [`wider_type`] for the elements and fields. A nested field
/// is nullable if it is on either side. Struct fields must have the same names in the same
/// order. Spark compares them ignoring case, but this requires an exact match.
fn complex_type(left: &DataType, right: &DataType, ansi_mode: bool) -> Option<DataType> {
    use DataType::*;
    match (left, right) {
        (
            List(lf) | LargeList(lf) | FixedSizeList(lf, _),
            List(rf) | LargeList(rf) | FixedSizeList(rf, _),
        ) => {
            let item = wider_type(lf.data_type(), rf.data_type(), ansi_mode)?;
            let field = Arc::new(Field::new(
                lf.name(),
                item,
                lf.is_nullable() || rf.is_nullable(),
            ));
            Some(match (left, right) {
                (FixedSizeList(_, ln), FixedSizeList(_, rn)) if ln == rn => {
                    FixedSizeList(field, *ln)
                }
                (LargeList(_), _) | (_, LargeList(_)) => LargeList(field),
                _ => List(field),
            })
        }
        (Struct(lf), Struct(rf)) if lf.len() == rf.len() => {
            let fields = lf
                .iter()
                .zip(rf.iter())
                .map(|(a, b)| {
                    if a.name() != b.name() {
                        return None;
                    }
                    let data_type = wider_type(a.data_type(), b.data_type(), ansi_mode)?;
                    Some(Field::new(
                        a.name(),
                        data_type,
                        a.is_nullable() || b.is_nullable(),
                    ))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Struct(Fields::from(fields)))
        }
        // Spark finds a common map type, which is then rejected as not orderable.
        (Map(lf, sorted), Map(rf, _)) => {
            let entries = wider_type(lf.data_type(), rf.data_type(), ansi_mode)?;
            Some(Map(
                Arc::new(Field::new(lf.name(), entries, false)),
                *sorted,
            ))
        }
        _ => None,
    }
}

/// Whether Spark can order values of this type. It cannot order maps, at any depth.
fn is_orderable(data_type: &DataType) -> bool {
    match data_type {
        DataType::Map(_, _) | DataType::Union(_, _) => false,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => is_orderable(field.data_type()),
        DataType::Struct(fields) => fields.iter().all(|f| is_orderable(f.data_type())),
        DataType::Dictionary(_, value) => is_orderable(value),
        _ => true,
    }
}

fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

fn is_binary(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView
    )
}

/// Of two physical layouts of the same Spark string or binary type, the one that can hold the
/// values of both.
fn wider_physical_type(left: &DataType, right: &DataType) -> DataType {
    use DataType::*;
    match (left, right) {
        (LargeUtf8, _) | (_, LargeUtf8) => LargeUtf8,
        (LargeBinary, _) | (_, LargeBinary) => LargeBinary,
        (Utf8View, _) | (_, Utf8View) => Utf8View,
        (BinaryView, _) | (_, BinaryView) => BinaryView,
        _ => left.clone(),
    }
}

/// The precision of Spark's `DecimalType.forType` for an integral type.
fn integral_digits(data_type: &DataType) -> Option<i16> {
    match data_type {
        DataType::Int8 => Some(3),
        DataType::Int16 => Some(5),
        DataType::Int32 => Some(10),
        DataType::Int64 => Some(20),
        _ => None,
    }
}

/// Spark's `numericPrecedence`, which leaves out decimals.
fn numeric_precedence(data_type: &DataType) -> Option<u8> {
    match data_type {
        DataType::Int8 => Some(0),
        DataType::Int16 => Some(1),
        DataType::Int32 => Some(2),
        DataType::Int64 => Some(3),
        DataType::Float32 => Some(4),
        DataType::Float64 => Some(5),
        _ => None,
    }
}

fn finer_unit(left: TimeUnit, right: TimeUnit) -> TimeUnit {
    let rank = |unit: TimeUnit| match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 2,
        TimeUnit::Nanosecond => 3,
    };
    if rank(left) >= rank(right) {
        left
    } else {
        right
    }
}

/// The name Spark's `toSQLType` gives the Spark type that an Arrow type maps to, for error
/// messages. Arrow types without a Spark counterpart use their Arrow name.
fn spark_sql_type(data_type: &DataType) -> String {
    use DataType::*;
    match data_type {
        Null => "VOID".to_string(),
        Boolean => "BOOLEAN".to_string(),
        Int8 => "TINYINT".to_string(),
        Int16 => "SMALLINT".to_string(),
        Int32 => "INT".to_string(),
        Int64 => "BIGINT".to_string(),
        Float32 => "FLOAT".to_string(),
        Float64 => "DOUBLE".to_string(),
        Decimal128(p, s) => format!("DECIMAL({p},{s})"),
        Utf8 | LargeUtf8 | Utf8View => "STRING".to_string(),
        Binary | LargeBinary | BinaryView => "BINARY".to_string(),
        Date32 => "DATE".to_string(),
        Timestamp(_, Some(_)) => "TIMESTAMP".to_string(),
        Timestamp(_, None) => "TIMESTAMP_NTZ".to_string(),
        List(field) | LargeList(field) | FixedSizeList(field, _) => {
            format!("ARRAY<{}>", spark_sql_type(field.data_type()))
        }
        Struct(fields) => format!(
            "STRUCT<{}>",
            fields
                .iter()
                .map(|f| format!(
                    "{}: {}{}",
                    f.name(),
                    spark_sql_type(f.data_type()),
                    if f.is_nullable() { "" } else { " NOT NULL" }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Map(entries, _) => match entries.data_type() {
            Struct(kv) if kv.len() == 2 => format!(
                "MAP<{}, {}>",
                spark_sql_type(kv[0].data_type()),
                spark_sql_type(kv[1].data_type())
            ),
            other => format!("MAP<{other}>"),
        },
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        AsArray, Float32Array, Float64Array, Int32Array, ListArray, StringArray,
        StructArray,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::{Float64Type, Int32Type};

    /// Values on which Spark's ordering and IEEE 754 total order disagree, with neighbors and a
    /// NULL. `NEGATIVE_NAN` is what arithmetic produces on x86-64.
    const NEGATIVE_NAN: f64 = f64::from_bits(0xfff8_0000_0000_0000);
    const PAYLOAD_NAN: f64 = f64::from_bits(0x7ff0_0000_0000_0001);
    const EDGE_VALUES: [Option<f64>; 10] = [
        Some(f64::NEG_INFINITY),
        Some(-1.0),
        Some(-0.0),
        Some(0.0),
        Some(1.0),
        Some(f64::INFINITY),
        Some(f64::NAN),
        Some(NEGATIVE_NAN),
        Some(PAYLOAD_NAN),
        None,
    ];

    /// `SQLOrderingUtil.compareDoubles`.
    fn compare_doubles(left: f64, right: f64) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        match (left.is_nan(), right.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            (false, true) => Less,
            _ => left.partial_cmp(&right).unwrap(),
        }
    }

    /// Spark's `Greatest.eval` and `Least.eval`, over doubles.
    fn spark_extreme(values: &[Option<f64>], greatest: bool) -> Option<f64> {
        values
            .iter()
            .flatten()
            .fold(None, |best, &value| match best {
                None => Some(value),
                Some(best) => {
                    let ordering = compare_doubles(value, best);
                    let replace = if greatest {
                        ordering.is_gt()
                    } else {
                        ordering.is_lt()
                    };
                    Some(if replace { value } else { best })
                }
            })
    }

    fn udf(greatest: bool) -> GreatestLeast {
        GreatestLeast::new(greatest, true)
    }

    fn invoke(
        greatest: bool,
        args: Vec<ColumnarValue>,
        rows: usize,
        return_type: DataType,
    ) -> Result<ColumnarValue> {
        let return_field: FieldRef = Arc::new(Field::new("result", return_type, true));
        udf(greatest).invoke(ScalarFunctionArgs {
            args,
            arg_fields: vec![],
            number_rows: rows,
            return_field,
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn bits(array: &ArrayRef) -> Vec<Option<u64>> {
        let array = array.as_primitive::<Float64Type>();
        (0..array.len())
            .map(|i| array.is_valid(i).then(|| array.value(i).to_bits()))
            .collect()
    }

    /// Every combination of three edge values, as columns and with the middle argument also as
    /// a constant. The combinations without a NULL run again on their own.
    #[test]
    fn floats_follow_spark_order() -> Result<()> {
        let all_rows: Vec<[Option<f64>; 3]> = EDGE_VALUES
            .iter()
            .flat_map(|&a| {
                EDGE_VALUES
                    .iter()
                    .flat_map(move |&b| EDGE_VALUES.iter().map(move |&c| [a, b, c]))
            })
            .collect();
        let without_nulls: Vec<_> = all_rows
            .iter()
            .copied()
            .filter(|row| row.iter().all(Option::is_some))
            .collect();
        for rows in [all_rows, without_nulls] {
            let column = |i: usize| -> ColumnarValue {
                ColumnarValue::Array(Arc::new(Float64Array::from(
                    rows.iter().map(|row| row[i]).collect::<Vec<_>>(),
                )))
            };
            for greatest in [true, false] {
                let args = vec![column(0), column(1), column(2)];
                let result = invoke(greatest, args, rows.len(), DataType::Float64)?
                    .into_array(rows.len())?;
                let expected: Vec<Option<u64>> = rows
                    .iter()
                    .map(|row| spark_extreme(row, greatest).map(f64::to_bits))
                    .collect();
                assert_eq!(bits(&result), expected, "greatest={greatest}");

                for middle in EDGE_VALUES {
                    let args = vec![
                        column(0),
                        ColumnarValue::Scalar(ScalarValue::Float64(middle)),
                        column(2),
                    ];
                    let result = invoke(greatest, args, rows.len(), DataType::Float64)?
                        .into_array(rows.len())?;
                    let expected: Vec<Option<u64>> = rows
                        .iter()
                        .map(|row| {
                            spark_extreme(&[row[0], middle, row[2]], greatest)
                                .map(f64::to_bits)
                        })
                        .collect();
                    assert_eq!(
                        bits(&result),
                        expected,
                        "middle={middle:?} greatest={greatest}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn float32_ties_keep_the_first_argument() -> Result<()> {
        let zero = ColumnarValue::Array(Arc::new(Float32Array::from(vec![0.0f32])));
        let negative_zero =
            ColumnarValue::Array(Arc::new(Float32Array::from(vec![-0.0f32])));
        for greatest in [true, false] {
            for (args, expected) in [
                (vec![negative_zero.clone(), zero.clone()], -0.0f32),
                (vec![zero.clone(), negative_zero.clone()], 0.0f32),
            ] {
                let result =
                    invoke(greatest, args, 1, DataType::Float32)?.into_array(1)?;
                let value = result
                    .as_primitive::<arrow::datatypes::Float32Type>()
                    .value(0);
                assert_eq!(value.to_bits(), expected.to_bits(), "greatest={greatest}");
            }
        }
        Ok(())
    }

    #[test]
    fn constants_only_return_a_constant() -> Result<()> {
        let scalar = |v: Option<f64>| ColumnarValue::Scalar(ScalarValue::Float64(v));
        for (greatest, args, expected) in [
            (
                true,
                vec![scalar(Some(-0.0)), scalar(Some(0.0))],
                Some(-0.0),
            ),
            (
                false,
                vec![scalar(Some(0.0)), scalar(Some(-0.0))],
                Some(0.0),
            ),
            (
                true,
                vec![
                    scalar(None),
                    scalar(Some(f64::NAN)),
                    scalar(Some(f64::INFINITY)),
                ],
                Some(f64::NAN),
            ),
            (false, vec![scalar(None), scalar(None)], None),
        ] {
            match invoke(greatest, args, 5, DataType::Float64)? {
                ColumnarValue::Scalar(ScalarValue::Float64(actual)) => {
                    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits))
                }
                other => panic!("expected a double scalar, got {other:?}"),
            }
        }
        Ok(())
    }

    #[test]
    fn integers_and_strings_with_nulls() -> Result<()> {
        let ints =
            |v: Vec<Option<i32>>| ColumnarValue::Array(Arc::new(Int32Array::from(v)));
        let args = vec![
            ints(vec![Some(1), None, None, Some(5)]),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(3))),
            ints(vec![Some(2), Some(7), None, None]),
        ];
        let result = invoke(true, args.clone(), 4, DataType::Int32)?.into_array(4)?;
        assert_eq!(
            result.as_ref(),
            &Int32Array::from(vec![Some(3), Some(7), Some(3), Some(5)])
        );
        let result = invoke(false, args, 4, DataType::Int32)?.into_array(4)?;
        assert_eq!(
            result.as_ref(),
            &Int32Array::from(vec![Some(1), Some(3), Some(3), Some(3)])
        );

        let strings =
            |v: Vec<Option<&str>>| ColumnarValue::Array(Arc::new(StringArray::from(v)));
        let args = vec![
            strings(vec![Some("abc"), None, Some("")]),
            strings(vec![Some("abd"), None, Some("a")]),
        ];
        let result = invoke(true, args.clone(), 3, DataType::Utf8)?.into_array(3)?;
        assert_eq!(
            result.as_ref(),
            &StringArray::from(vec![Some("abd"), None, Some("a")])
        );
        let result = invoke(false, args, 3, DataType::Utf8)?.into_array(3)?;
        assert_eq!(
            result.as_ref(),
            &StringArray::from(vec![Some("abc"), None, Some("")])
        );
        Ok(())
    }

    /// A NULL inside a list ranks first, lists compare element by element and then by length,
    /// and floats inside lists follow Spark's order. The second list's element field has a
    /// different name and nullability, which the result type decides.
    #[test]
    fn nested_arguments() -> Result<()> {
        let list = |values: Vec<Option<Vec<Option<i32>>>>| -> ColumnarValue {
            ColumnarValue::Array(Arc::new(ListArray::from_iter_primitive::<
                Int32Type,
                _,
                _,
            >(values)))
        };
        let args = vec![
            list(vec![
                Some(vec![Some(1), None]),
                Some(vec![Some(1), Some(2)]),
                None,
            ]),
            list(vec![
                Some(vec![Some(1), Some(0)]),
                Some(vec![Some(1), Some(2), Some(3)]),
                None,
            ]),
        ];
        let list_type = args[0].data_type();
        for (greatest, expected) in [
            (
                true,
                list(vec![
                    Some(vec![Some(1), Some(0)]),
                    Some(vec![Some(1), Some(2), Some(3)]),
                    None,
                ]),
            ),
            (
                false,
                list(vec![
                    Some(vec![Some(1), None]),
                    Some(vec![Some(1), Some(2)]),
                    None,
                ]),
            ),
        ] {
            let result =
                invoke(greatest, args.clone(), 3, list_type.clone())?.into_array(3)?;
            assert_eq!(result.as_ref(), expected.into_array(3)?.as_ref());
        }

        let float_list =
            |name: &str, nullable: bool, values: Vec<f64>| -> ColumnarValue {
                let field = Arc::new(Field::new(name, DataType::Float64, nullable));
                ColumnarValue::Array(Arc::new(ListArray::new(
                    field,
                    OffsetBuffer::from_lengths(vec![1; values.len()]),
                    Arc::new(Float64Array::from(values)),
                    None,
                )))
            };
        let return_type =
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true)));
        for (greatest, expected) in
            [(true, [-0.0, f64::NAN]), (false, [-0.0, f64::INFINITY])]
        {
            let args = vec![
                float_list("item", true, vec![-0.0, f64::INFINITY]),
                float_list("element", false, vec![0.0, f64::NAN]),
            ];
            let result = invoke(greatest, args, 2, return_type.clone())?.into_array(2)?;
            assert_eq!(result.data_type(), &return_type);
            let values: ArrayRef = Arc::clone(result.as_list::<i32>().values());
            let expected: Vec<Option<u64>> =
                expected.iter().map(|v| Some(v.to_bits())).collect();
            assert_eq!(bits(&values), expected, "greatest={greatest}");
        }

        // A constant struct against a column of structs.
        let fields = Fields::from(vec![Field::new("a", DataType::Float64, true)]);
        let structs = |values: Vec<Option<f64>>| -> ArrayRef {
            Arc::new(StructArray::new(
                fields.clone(),
                vec![Arc::new(Float64Array::from(values))],
                None,
            ))
        };
        let constant = ScalarValue::try_from_array(&structs(vec![Some(f64::NAN)]), 0)?;
        let args = vec![
            ColumnarValue::Array(structs(vec![Some(1.0), None])),
            ColumnarValue::Scalar(constant),
        ];
        let struct_type = DataType::Struct(fields.clone());
        let result = invoke(true, args.clone(), 2, struct_type.clone())?.into_array(2)?;
        let a: ArrayRef = Arc::clone(result.as_struct().column(0));
        assert_eq!(
            bits(&a),
            vec![Some(f64::NAN.to_bits()), Some(f64::NAN.to_bits())]
        );
        let result = invoke(false, args, 2, struct_type)?.into_array(2)?;
        let a: ArrayRef = Arc::clone(result.as_struct().column(0));
        // A NULL field ranks first.
        assert_eq!(bits(&a), vec![Some(1.0f64.to_bits()), None]);
        Ok(())
    }

    #[test]
    fn null_type() -> Result<()> {
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Null),
            ColumnarValue::Scalar(ScalarValue::Null),
        ];
        assert!(matches!(
            invoke(true, args, 3, DataType::Null)?,
            ColumnarValue::Scalar(ScalarValue::Null)
        ));
        Ok(())
    }

    #[test]
    fn rejects_fewer_than_two_arguments() {
        let scalar = ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0)));
        for greatest in [true, false] {
            for arg_types in [vec![], vec![DataType::Float64]] {
                let error = udf(greatest).coerce_types(&arg_types).unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("[WRONG_NUM_ARGS.WITHOUT_SUGGESTION]"),
                    "unexpected error: {error}"
                );
            }
            let error =
                invoke(greatest, vec![scalar.clone()], 1, DataType::Float64).unwrap_err();
            assert!(
                error.to_string().contains("requires > 1 parameters"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn coercion() {
        use DataType::*;
        let coerce = |types: Vec<DataType>, ansi: bool| {
            GreatestLeast::new(true, ansi)
                .coerce_types(&types)
                .map(|t| t[0].clone())
        };
        let list = |t: DataType| List(Arc::new(Field::new("item", t, true)));
        let ok = [
            (vec![Int32, Float64], true, Float64),
            (vec![Int32, Float32], true, Float64),
            (vec![Int64, Float32], false, Float32),
            (vec![Int8, Int16], true, Int16),
            (vec![Null, Int32, Null], true, Int32),
            (vec![Null, Null], true, Null),
            (vec![Decimal128(11, 1), Int32], true, Decimal128(11, 1)),
            (vec![Decimal128(2, 1), Int32], true, Decimal128(11, 1)),
            (vec![Decimal128(5, 2), Int64], true, Decimal128(22, 2)),
            (
                vec![Decimal128(38, 10), Decimal128(38, 0)],
                true,
                Decimal128(38, 0),
            ),
            (vec![Decimal128(10, 2), Float32], true, Float64),
            (vec![Utf8, Utf8View], true, Utf8View),
            (
                vec![Date32, Timestamp(TimeUnit::Microsecond, Some("UTC".into()))],
                true,
                Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            ),
            (vec![list(Int32), list(Float64)], true, list(Float64)),
        ];
        for (types, ansi, expected) in ok {
            assert_eq!(coerce(types.clone(), ansi).unwrap(), expected, "{types:?}");
        }
        let diff_types = [
            vec![Int32, Utf8],
            vec![Date32, Utf8],
            vec![Boolean, Int32],
            vec![list(Int32), Int32],
        ];
        for types in diff_types {
            let error = coerce(types.clone(), true).unwrap_err().to_string();
            assert!(
                error.contains("[DATATYPE_MISMATCH.DATA_DIFF_TYPES]"),
                "{types:?}: {error}"
            );
        }
        let error = coerce(vec![Int32, Utf8], true).unwrap_err().to_string();
        assert!(error.contains(r#"but it's ["INT", "STRING"]"#), "{error}");

        let map = Map(
            Arc::new(Field::new(
                "entries",
                Struct(Fields::from(vec![
                    Field::new("key", Int32, false),
                    Field::new("value", Int32, true),
                ])),
                false,
            )),
            false,
        );
        let error = coerce(vec![map.clone(), map], true)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(r#"[DATATYPE_MISMATCH.INVALID_ORDERING_TYPE] The `greatest` does not support ordering on type "MAP<INT, INT>""#),
            "{error}"
        );
    }
}
