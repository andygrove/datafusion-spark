// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Casting a function's arguments to the input types it declares, as Spark does.
//!
//! Most Spark expressions mix in `ImplicitCastInputTypes` and declare
//! `inputTypes: Seq[AbstractDataType]`. The analyzer then casts each argument with
//! `implicitCast` from `TypeCoercion` (ANSI mode off) or `AnsiTypeCoercion` (ANSI mode on),
//! and `ExpectsInputTypes.checkInputDataTypes` raises `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE`
//! for an argument that cannot be cast.
//!
//! A function here declares the same types with [`SparkInputType`], holds them in a
//! [`SparkInputTypes`], and takes its signature from it:
//!
//! ```ignore
//! fn signature(&self) -> &Signature {
//!     self.input_types.signature()
//! }
//!
//! fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
//!     self.input_types.coerce(self.name(), arg_types)
//! }
//!
//! fn with_updated_config(&self, config: &ConfigOptions) -> Option<ScalarUDF> {
//!     let input_types = self.input_types.with_config(config)?;
//!     Some(ScalarUDF::from(Self { input_types }))
//! }
//! ```
//!
//! This module only chooses the target types. The cast that runs is DataFusion's, so how a
//! malformed string or an overflowing value is cast is part of the CAST work in #88.
//!
//! For expressions that use only `ExpectsInputTypes`, construct the declarations with
//! [`SparkInputTypes::check_only`]. This shares the validation and error messages without
//! accepting implicit casts. Functions may then widen validated inputs for execution.

use arrow::datatypes::{DataType, IntervalUnit};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, plan_err};
use datafusion_expr::{Signature, Volatility};

/// A type that a Spark function declares for one of its arguments: a concrete Spark type,
/// or one of Spark's abstract types such as `NumericType`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[allow(
    dead_code,
    reason = "declared for the functions that move onto these rules after sha2 (#346)"
)]
pub(crate) enum SparkInputType {
    /// `AnyDataType`: accepts every type and never casts.
    Any,
    /// `BooleanType`
    Boolean,
    /// `ByteType` (TINYINT)
    Byte,
    /// `ShortType` (SMALLINT)
    Short,
    /// `IntegerType` (INT)
    Integer,
    /// `LongType` (BIGINT)
    Long,
    /// `FloatType`
    Float,
    /// `DoubleType`
    Double,
    /// The abstract `NumericType`. Its default type is DOUBLE.
    Numeric,
    /// The abstract `IntegralType`. Its default type is INT.
    Integral,
    /// The abstract `DecimalType`: a decimal of any precision and scale. Its default type is
    /// `DECIMAL(38,18)`.
    Decimal,
    /// `StringType`
    String,
    /// `BinaryType`
    Binary,
    /// `AbstractArrayType(element)`: an array whose elements are cast to `element`.
    Array(Box<SparkInputType>),
    /// `TypeCollection(types)`: the first type that the argument is accepted as, or else the
    /// first that it can be cast to.
    OneOf(Vec<SparkInputType>),
}

/// The input types a function declares, whether implicit casts are allowed, and the ANSI
/// mode that controls those casts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct SparkInputTypes {
    types: Vec<SparkInputType>,
    ansi_mode: bool,
    implicit_casts: bool,
    signature: Signature,
}

impl SparkInputTypes {
    /// Declares one input type per argument, with ANSI mode off.
    pub(crate) fn new(types: Vec<SparkInputType>, volatility: Volatility) -> Self {
        Self {
            types,
            ansi_mode: false,
            implicit_casts: true,
            signature: Signature::user_defined(volatility),
        }
    }

    /// Declares `ExpectsInputTypes`: validate inputs without implicit casts.
    /// NULL uses the declared type's default, and dictionary layouts are unwrapped.
    pub(crate) fn check_only(types: Vec<SparkInputType>, volatility: Volatility) -> Self {
        Self {
            implicit_casts: false,
            ..Self::new(types, volatility)
        }
    }

    /// The function's `Signature`, which leaves coercion to [`Self::coerce`].
    pub(crate) fn signature(&self) -> &Signature {
        &self.signature
    }

    /// Returns the input types for `config`, or `None` when ANSI mode is unchanged.
    ///
    /// Call it from `ScalarUDFImpl::with_updated_config`, which is how a function sees the
    /// session's `datafusion.execution.enable_ansi_mode`.
    pub(crate) fn with_config(&self, config: &ConfigOptions) -> Option<Self> {
        let ansi_mode = config.execution.enable_ansi_mode;
        (ansi_mode != self.ansi_mode).then(|| Self {
            ansi_mode,
            ..self.clone()
        })
    }

    /// Chooses each argument's target type for `ScalarUDFImpl::coerce_types`.
    /// In check-only mode, only accepted layouts and untyped NULLs are allowed.
    ///
    /// Returns Spark's `WRONG_NUM_ARGS` error for the wrong number of arguments, and
    /// `DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE` for an argument that cannot be cast.
    pub(crate) fn coerce(
        &self,
        name: &str,
        arg_types: &[DataType],
    ) -> Result<Vec<DataType>> {
        if arg_types.len() != self.types.len() {
            return plan_err!(
                "[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `{name}` requires {} parameters but \
                 the actual number is {}. Please, refer to \
                 'https://spark.apache.org/docs/latest/sql-ref-functions.html' for a fix. \
                 SQLSTATE: 42605",
                self.types.len(),
                arg_types.len()
            );
        }
        arg_types
            .iter()
            .zip(&self.types)
            .enumerate()
            .map(|(idx, (arg_type, expected))| {
                let target = if self.implicit_casts {
                    implicit_cast(arg_type, expected, self.ansi_mode)
                } else {
                    let layout = spark_layout(arg_type);
                    if expected.accepts(&layout) {
                        Some(layout)
                    } else if layout == DataType::Null {
                        expected.default_type()
                    } else {
                        None
                    }
                };
                match target {
                    Some(target) => Ok(target),
                    None => plan_err!(
                        "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The {} parameter of \
                         `{name}` requires the {} type, however it has the type \"{}\". \
                         SQLSTATE: 42K09",
                        ordinal_number(idx),
                        expected.sql(),
                        spark_sql_type(arg_type)
                    ),
                }
            })
            .collect()
    }
}

impl SparkInputType {
    /// Spark's `AbstractDataType.acceptsType`: whether an argument of this type is used as it
    /// is, without a cast.
    fn accepts(&self, data_type: &DataType) -> bool {
        use SparkInputType::*;
        match self {
            Any => true,
            Boolean => matches!(data_type, DataType::Boolean),
            Byte => matches!(data_type, DataType::Int8),
            Short => matches!(data_type, DataType::Int16),
            Integer => matches!(data_type, DataType::Int32),
            Long => matches!(data_type, DataType::Int64),
            Float => matches!(data_type, DataType::Float32),
            Double => matches!(data_type, DataType::Float64),
            Numeric => data_type.is_numeric(),
            Integral => data_type.is_integer(),
            Decimal => data_type.is_decimal(),
            String => data_type.is_string(),
            Binary => data_type.is_binary(),
            Array(element) => {
                list_element(data_type).is_some_and(|item| element.accepts(item))
            }
            OneOf(types) => types.iter().any(|t| t.accepts(data_type)),
        }
    }

    /// Spark's `AbstractDataType.defaultConcreteType`: the type an argument is cast to when
    /// the declared type is abstract. `None` for [`SparkInputType::Any`], which never casts.
    fn default_type(&self) -> Option<DataType> {
        use SparkInputType::*;
        Some(match self {
            Any => return None,
            Boolean => DataType::Boolean,
            Byte => DataType::Int8,
            Short => DataType::Int16,
            Integer | Integral => DataType::Int32,
            Long => DataType::Int64,
            Float => DataType::Float32,
            Double | Numeric => DataType::Float64,
            Decimal => DataType::Decimal128(38, 18),
            String => DataType::Utf8,
            Binary => DataType::Binary,
            Array(element) => DataType::new_list(element.default_type()?, true),
            OneOf(types) => return types.first()?.default_type(),
        })
    }

    /// Whether this is a concrete Spark `AtomicType` (and not an abstract type).
    fn is_concrete_atomic(&self) -> bool {
        use SparkInputType::*;
        self.is_concrete_numeric() || matches!(self, Boolean | String | Binary)
    }

    /// Whether this is a concrete Spark `NumericType` (and not `Numeric` or `Decimal`).
    fn is_concrete_numeric(&self) -> bool {
        use SparkInputType::*;
        matches!(self, Byte | Short | Integer | Long | Float | Double)
    }

    /// The type as Spark's `toSQLType` writes it in an error message, with the quotes.
    fn sql(&self) -> String {
        use SparkInputType::*;
        match self {
            OneOf(types) => format!(
                "({})",
                types
                    .iter()
                    .map(|t| t.sql())
                    .collect::<Vec<_>>()
                    .join(" or ")
            ),
            other => format!("\"{}\"", other.simple_string()),
        }
    }

    fn simple_string(&self) -> String {
        use SparkInputType::*;
        match self {
            Any => "ANY",
            Boolean => "BOOLEAN",
            Byte => "TINYINT",
            Short => "SMALLINT",
            Integer => "INT",
            Long => "BIGINT",
            Float => "FLOAT",
            Double => "DOUBLE",
            Numeric => "NUMERIC",
            Integral => "INTEGRAL",
            Decimal => "DECIMAL",
            String => "STRING",
            Binary => "BINARY",
            Array(element) => return format!("ARRAY<{}>", element.simple_string()),
            // `toSQLType` upper-cases the whole `simpleString` of an abstract type, so a
            // collection inside an array reads `ARRAY<(STRING OR BINARY)>`.
            OneOf(types) => {
                return format!(
                    "({})",
                    types
                        .iter()
                        .map(|t| t.simple_string())
                        .collect::<Vec<_>>()
                        .join(" OR ")
                );
            }
        }
        .to_string()
    }
}

/// The type to cast an argument of type `from` to, for a declared `expected` type, or `None`
/// if Spark does not cast it.
///
/// A port of the private `implicitCast(DataType, AbstractDataType)` in Spark's
/// `TypeCoercion` (ANSI mode off) and `AnsiTypeCoercion` (ANSI mode on).
fn implicit_cast(
    from: &DataType,
    expected: &SparkInputType,
    ansi_mode: bool,
) -> Option<DataType> {
    let from = &spark_layout(from);
    if expected.accepts(from) {
        return Some(from.clone());
    }
    if ansi_mode {
        ansi_implicit_cast(from, expected)
    } else {
        legacy_implicit_cast(from, expected)
    }
}

/// The layout a function receives for an argument of a type Spark does not have.
///
/// A dictionary has the type of its values (see #349), and a fixed-size binary becomes
/// BINARY, so that a function only has to handle the layouts of Spark's own types.
fn spark_layout(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value) => spark_layout(value),
        DataType::FixedSizeBinary(_) => DataType::Binary,
        other => other.clone(),
    }
}

/// `TypeCoercion.implicitCast`, with ANSI mode off.
fn legacy_implicit_cast(from: &DataType, expected: &SparkInputType) -> Option<DataType> {
    use SparkInputType as T;
    match (from, expected) {
        (DataType::Null, target) => target.default_type(),
        // Hive's convention: a string where any number is accepted becomes a DOUBLE.
        (f, T::Numeric) if f.is_string() => Some(DataType::Float64),
        (f, T::Decimal) if f.is_numeric() => decimal_for_type(f),
        (f, target) if f.is_numeric() && target.is_concrete_numeric() => {
            target.default_type()
        }
        (f, T::Decimal) if f.is_string() => T::Decimal.default_type(),
        (f, target) if f.is_string() && target.is_concrete_numeric() => {
            target.default_type()
        }
        (f, T::Binary) if f.is_string() => Some(binary_for_string(f)),
        (f, T::String) if is_atomic(f) => Some(DataType::Utf8),
        (f, T::OneOf(types)) => types.iter().find_map(|t| implicit_cast(f, t, false)),
        (f, T::Array(element)) => {
            let item = list_element(f)?;
            implicit_cast(item, element, false).map(|t| DataType::new_list(t, true))
        }
        _ => None,
    }
}

/// `AnsiTypeCoercion.implicitCast`, with ANSI mode on.
fn ansi_implicit_cast(from: &DataType, expected: &SparkInputType) -> Option<DataType> {
    use SparkInputType as T;
    match (from, expected) {
        (f, T::Decimal) if f.is_numeric() => decimal_for_type(f),
        (f, T::Integral) if is_fractional(f) => None,
        // Spark allows a string to be cast to any primitive type, to keep existing queries
        // working, although `Cast.canANSIStoreAssign` does not.
        (f, T::Binary) if f.is_string() => Some(binary_for_string(f)),
        (f, target)
            if f.is_string()
                && (target.is_concrete_atomic()
                    || matches!(target, T::Numeric | T::Decimal)) =>
        {
            target.default_type()
        }
        (f, T::Array(element)) if list_element(f).is_some() => {
            let item = list_element(f)?;
            implicit_cast(item, element, true).map(|t| DataType::new_list(t, true))
        }
        (f, T::OneOf(types)) => types.iter().find_map(|t| implicit_cast(f, t, true)),
        (f, target) => {
            let concrete = target.default_type()?;
            can_ansi_store_assign(f, &concrete).then_some(concrete)
        }
    }
}

/// Spark's `Cast.canANSIStoreAssign`: whether a value of type `from` can be stored in a
/// column of type `to` under ANSI rules.
fn can_ansi_store_assign(from: &DataType, to: &DataType) -> bool {
    if let (Some(from_item), Some(to_item)) = (list_element(from), list_element(to)) {
        return can_ansi_store_assign(from_item, to_item);
    }
    match (from, to) {
        _ if from == to => true,
        (DataType::Null, _) => true,
        (f, t) if f.is_numeric() && t.is_numeric() => true,
        (f, t) if t.is_string() && (is_atomic(f) || is_calendar_interval(f)) => true,
        (f, t) if is_datetime(f) && is_datetime(t) => true,
        (DataType::Map(from_entries, _), DataType::Map(to_entries, _)) => {
            can_ansi_store_assign(from_entries.data_type(), to_entries.data_type())
        }
        (DataType::Struct(from_fields), DataType::Struct(to_fields)) => {
            from_fields.len() == to_fields.len()
                && from_fields.iter().zip(to_fields.iter()).all(|(f, t)| {
                    (t.is_nullable() || !f.is_nullable())
                        && can_ansi_store_assign(f.data_type(), t.data_type())
                })
        }
        _ => false,
    }
}

/// Spark's `DecimalType.forType`: the decimal that holds every value of a numeric type.
fn decimal_for_type(data_type: &DataType) -> Option<DataType> {
    use DataType::*;
    let (precision, scale) = match data_type {
        Int8 | UInt8 => (3, 0),
        Int16 | UInt16 => (5, 0),
        Int32 | UInt32 => (10, 0),
        Int64 | UInt64 => (20, 0),
        Float16 | Float32 => (14, 7),
        Float64 => (30, 15),
        _ => return None,
    };
    Some(Decimal128(precision, scale))
}

/// The binary layout that matches a string layout, so the cast keeps the buffers.
fn binary_for_string(data_type: &DataType) -> DataType {
    match data_type {
        DataType::LargeUtf8 => DataType::LargeBinary,
        DataType::Utf8View => DataType::BinaryView,
        _ => DataType::Binary,
    }
}

fn list_element(data_type: &DataType) -> Option<&DataType> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _) => Some(field.data_type()),
        _ => None,
    }
}

/// Spark's `FractionalType`: FLOAT, DOUBLE and DECIMAL.
fn is_fractional(data_type: &DataType) -> bool {
    data_type.is_floating() || data_type.is_decimal()
}

/// Spark's `DatetimeType`: DATE, TIMESTAMP, TIMESTAMP_NTZ and TIME.
fn is_datetime(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
            | DataType::Time32(_)
            | DataType::Time64(_)
    )
}

/// Spark's `CalendarIntervalType`, which is not an `AtomicType`.
fn is_calendar_interval(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Interval(IntervalUnit::MonthDayNano))
}

/// Spark's `AtomicType`: every type except NULL, arrays, maps, structs and
/// `CalendarIntervalType`. The year-month and day-time interval types are atomic.
fn is_atomic(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Interval(IntervalUnit::YearMonth | IntervalUnit::DayTime)
            | DataType::Duration(_)
    ) || data_type.is_numeric()
        || data_type.is_string()
        || data_type.is_binary()
        || is_datetime(data_type)
}

/// Spark's `ordinalNumber`, for "The first parameter ...". Spark spells out only the first
/// three.
fn ordinal_number(idx: usize) -> String {
    match idx {
        0 => "first".to_string(),
        1 => "second".to_string(),
        2 => "third".to_string(),
        _ => format!("{}th", idx + 1),
    }
}

/// The name Spark's `toSQLType` gives a type, for error messages.
///
/// The same function is in `type_coercion.rs` in #345; whichever lands second should use the
/// other's copy.
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
        Interval(IntervalUnit::MonthDayNano) => "INTERVAL".to_string(),
        Interval(IntervalUnit::YearMonth) => "INTERVAL YEAR TO MONTH".to_string(),
        Duration(_) => "INTERVAL DAY TO SECOND".to_string(),
        // Arrow has no TIME precision. Spark's `make_time` returns TIME(6).
        Time64(_) => "TIME(6)".to_string(),
        List(field) | LargeList(field) | FixedSizeList(field, _) => {
            format!("ARRAY<{}>", spark_sql_type(field.data_type()))
        }
        Struct(fields) => format!(
            "STRUCT<{}>",
            fields
                .iter()
                .map(|f| format!(
                    "{}: {}{}",
                    quote_if_needed(f.name()),
                    spark_sql_type(f.data_type()),
                    if f.is_nullable() { "" } else { " NOT NULL" }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Map(entries, _) => match entries.data_type() {
            Struct(kv) if let [k, v] = &kv[..] => format!(
                "MAP<{}, {}>",
                spark_sql_type(k.data_type()),
                spark_sql_type(v.data_type())
            ),
            other => format!("MAP<{other}>"),
        },
        Dictionary(_, value) => spark_sql_type(value),
        other => other.to_string(),
    }
}

/// Spark's `QuotingUtils.quoteIfNeeded`, which `StructField.sql` uses for field names.
fn quote_if_needed(name: &str) -> String {
    let mut chars = name.chars();
    let plain = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        name.to_string()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

#[cfg(test)]
mod tests {
    use super::SparkInputType as T;
    use super::*;

    fn cast(from: DataType, expected: T) -> (Option<DataType>, Option<DataType>) {
        (
            implicit_cast(&from, &expected, false),
            implicit_cast(&from, &expected, true),
        )
    }

    fn both(t: DataType) -> (Option<DataType>, Option<DataType>) {
        (Some(t.clone()), Some(t))
    }

    fn list(item: DataType) -> DataType {
        DataType::new_list(item, true)
    }

    #[test]
    fn accepted_types_are_not_cast() {
        assert_eq!(cast(DataType::Int32, T::Integer), both(DataType::Int32));
        assert_eq!(
            cast(DataType::LargeBinary, T::Binary),
            both(DataType::LargeBinary)
        );
        assert_eq!(
            cast(DataType::Utf8View, T::String),
            both(DataType::Utf8View)
        );
        assert_eq!(
            cast(DataType::Decimal128(10, 2), T::Numeric),
            both(DataType::Decimal128(10, 2))
        );
        assert_eq!(cast(DataType::Int8, T::Integral), both(DataType::Int8));
        assert_eq!(
            cast(list(DataType::Int32), T::Any),
            both(list(DataType::Int32))
        );
    }

    #[test]
    fn null_becomes_the_default_type() {
        assert_eq!(cast(DataType::Null, T::Binary), both(DataType::Binary));
        assert_eq!(cast(DataType::Null, T::Numeric), both(DataType::Float64));
        assert_eq!(
            cast(DataType::Null, T::OneOf(vec![T::String, T::Binary])),
            both(DataType::Utf8)
        );
    }

    #[test]
    fn numbers_cast_to_each_other() {
        assert_eq!(cast(DataType::Int64, T::Integer), both(DataType::Int32));
        assert_eq!(cast(DataType::Float64, T::Integer), both(DataType::Int32));
        assert_eq!(
            cast(DataType::Decimal128(10, 2), T::Integer),
            both(DataType::Int32)
        );
        assert_eq!(cast(DataType::Int32, T::Double), both(DataType::Float64));
        assert_eq!(cast(DataType::UInt16, T::Integer), both(DataType::Int32));
    }

    #[test]
    fn numbers_cast_to_the_decimal_that_holds_them() {
        assert_eq!(
            cast(DataType::Int32, T::Decimal),
            both(DataType::Decimal128(10, 0))
        );
        assert_eq!(
            cast(DataType::Int64, T::Decimal),
            both(DataType::Decimal128(20, 0))
        );
        assert_eq!(
            cast(DataType::Float64, T::Decimal),
            both(DataType::Decimal128(30, 15))
        );
        assert_eq!(
            cast(DataType::Utf8, T::Decimal),
            both(DataType::Decimal128(38, 18))
        );
    }

    #[test]
    fn strings_cast_to_numbers_and_binary() {
        assert_eq!(cast(DataType::Utf8, T::Numeric), both(DataType::Float64));
        assert_eq!(cast(DataType::Utf8, T::Integer), both(DataType::Int32));
        assert_eq!(cast(DataType::Utf8, T::Binary), both(DataType::Binary));
        assert_eq!(
            cast(DataType::LargeUtf8, T::Binary),
            both(DataType::LargeBinary)
        );
        assert_eq!(
            cast(DataType::Utf8View, T::Binary),
            both(DataType::BinaryView)
        );
    }

    #[test]
    fn atomic_types_cast_to_string() {
        assert_eq!(cast(DataType::Int32, T::String), both(DataType::Utf8));
        assert_eq!(cast(DataType::Boolean, T::String), both(DataType::Utf8));
        assert_eq!(cast(DataType::Date32, T::String), both(DataType::Utf8));
        assert_eq!(cast(DataType::Binary, T::String), both(DataType::Utf8));
        assert_eq!(cast(list(DataType::Utf8), T::String), (None, None));
    }

    #[test]
    fn types_that_are_never_cast() {
        assert_eq!(cast(DataType::Int32, T::Binary), (None, None));
        assert_eq!(cast(DataType::Boolean, T::Integer), (None, None));
        assert_eq!(cast(DataType::Date32, T::Integer), (None, None));
        assert_eq!(cast(list(DataType::Utf8), T::Binary), (None, None));
        assert_eq!(cast(DataType::Utf8, T::Integral), (None, None));
    }

    #[test]
    fn ansi_mode_changes_some_casts() {
        // A string becomes a BOOLEAN only with ANSI mode on.
        assert_eq!(
            cast(DataType::Utf8, T::Boolean),
            (None, Some(DataType::Boolean))
        );
        // Neither mode casts a fraction to INTEGRAL: the legacy rules cast only to concrete
        // numeric types, and the ANSI rules reject fractions outright.
        assert_eq!(cast(DataType::Decimal128(10, 2), T::Integral), (None, None));
        assert_eq!(cast(DataType::Float64, T::Integral), (None, None));
        assert_eq!(cast(DataType::Int64, T::Integral), both(DataType::Int64));
        // A CalendarInterval becomes a STRING only with ANSI mode on.
        assert_eq!(
            cast(DataType::Interval(IntervalUnit::MonthDayNano), T::String),
            (None, Some(DataType::Utf8))
        );
    }

    #[test]
    fn one_of_picks_the_first_type_that_works() {
        let string_or_binary = T::OneOf(vec![T::String, T::Binary]);
        assert_eq!(
            cast(DataType::Binary, string_or_binary.clone()),
            both(DataType::Binary)
        );
        assert_eq!(
            cast(DataType::Int32, string_or_binary),
            both(DataType::Utf8)
        );
    }

    #[test]
    fn arrays_cast_element_by_element() {
        let strings = T::Array(Box::new(T::String));
        assert_eq!(
            cast(list(DataType::Int32), strings.clone()),
            both(list(DataType::Utf8))
        );
        assert_eq!(cast(list(list(DataType::Int32)), strings), (None, None));
    }

    #[test]
    fn dictionaries_have_the_type_of_their_values() {
        let dict =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        assert_eq!(cast(dict.clone(), T::String), both(DataType::Utf8));
        assert_eq!(cast(dict, T::Binary), both(DataType::Binary));
    }

    #[test]
    fn fixed_size_binary_becomes_binary() {
        assert_eq!(
            cast(DataType::FixedSizeBinary(5), T::Binary),
            both(DataType::Binary)
        );
        assert_eq!(
            cast(DataType::FixedSizeBinary(5), T::String),
            both(DataType::Utf8)
        );
    }

    #[test]
    fn coerce_reports_spark_errors() {
        let types =
            SparkInputTypes::new(vec![T::Binary, T::Integer], Volatility::Immutable);
        assert_eq!(
            types
                .coerce("sha2", &[DataType::Utf8, DataType::Int64])
                .unwrap(),
            vec![DataType::Binary, DataType::Int32]
        );

        let err = types
            .coerce("sha2", &[DataType::Int32, DataType::Int32])
            .unwrap_err()
            .strip_backtrace();
        assert!(
            err.contains(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `sha2` \
                 requires the \"BINARY\" type, however it has the type \"INT\". \
                 SQLSTATE: 42K09"
            ),
            "{err}"
        );

        let err = types
            .coerce("sha2", &[DataType::Utf8])
            .unwrap_err()
            .strip_backtrace();
        assert!(
            err.contains(
                "[WRONG_NUM_ARGS.WITHOUT_SUGGESTION] The `sha2` requires 2 parameters but \
                 the actual number is 1."
            ),
            "{err}"
        );

        let one_of = SparkInputTypes::new(
            vec![T::OneOf(vec![T::String, T::Binary])],
            Volatility::Immutable,
        );
        let err = one_of
            .coerce("f", &[list(DataType::Int32)])
            .unwrap_err()
            .strip_backtrace();
        assert!(
            err.contains("requires the (\"STRING\" or \"BINARY\") type"),
            "{err}"
        );
    }

    #[test]
    fn check_only_validates_without_implicit_casts() {
        let types = SparkInputTypes::check_only(vec![T::Integral], Volatility::Stable);
        for ansi_mode in [false, true] {
            let mut config = ConfigOptions::default();
            config.execution.enable_ansi_mode = ansi_mode;
            let types = types.with_config(&config).unwrap_or_else(|| types.clone());
            for input in [DataType::Int8, DataType::Int64, DataType::UInt32] {
                assert_eq!(
                    types
                        .coerce("timestamp_micros", std::slice::from_ref(&input))
                        .unwrap(),
                    vec![input]
                );
            }
            assert_eq!(
                types.coerce("timestamp_micros", &[DataType::Null]).unwrap(),
                vec![DataType::Int32]
            );
            assert_eq!(
                types
                    .coerce(
                        "timestamp_micros",
                        &[DataType::Dictionary(
                            Box::new(DataType::Int32),
                            Box::new(DataType::UInt32)
                        )]
                    )
                    .unwrap(),
                vec![DataType::UInt32]
            );
            for input in [
                DataType::Utf8,
                DataType::Float64,
                DataType::Decimal128(10, 1),
                DataType::Boolean,
            ] {
                let error = types
                    .coerce("timestamp_micros", &[input])
                    .unwrap_err()
                    .strip_backtrace();
                assert!(
                    error.contains(
                        "The first parameter of `timestamp_micros` requires the \"INTEGRAL\" type"
                    ),
                    "{error}"
                );
            }
            assert!(
                types
                    .coerce("timestamp_micros", &[])
                    .unwrap_err()
                    .strip_backtrace()
                    .contains("WRONG_NUM_ARGS")
            );
        }
        // Concrete types demonstrate that validation does not cast a string or
        // narrow an integer, unlike ImplicitCastInputTypes.
        let types = SparkInputTypes::check_only(vec![T::Integer], Volatility::Immutable);
        assert!(types.coerce("f", &[DataType::Utf8]).is_err());
        assert!(types.coerce("f", &[DataType::Int64]).is_err());
    }

    #[test]
    fn with_config_follows_ansi_mode() {
        let types = SparkInputTypes::new(vec![T::Boolean], Volatility::Immutable);
        let mut config = ConfigOptions::default();
        assert!(types.with_config(&config).is_none());

        config.execution.enable_ansi_mode = true;
        let ansi = types.with_config(&config).unwrap();
        assert!(types.coerce("f", &[DataType::Utf8]).is_err());
        assert_eq!(
            ansi.coerce("f", &[DataType::Utf8]).unwrap(),
            vec![DataType::Boolean]
        );
    }

    #[test]
    fn ordinals() {
        assert_eq!(ordinal_number(0), "first");
        assert_eq!(ordinal_number(2), "third");
        assert_eq!(ordinal_number(3), "4th");
        assert_eq!(ordinal_number(10), "11th");
    }

    #[test]
    fn type_names_match_to_sql_type() {
        let string_or_binary = T::OneOf(vec![T::String, T::Binary]);
        assert_eq!(string_or_binary.sql(), "(\"STRING\" or \"BINARY\")");
        assert_eq!(
            T::Array(Box::new(string_or_binary)).sql(),
            "\"ARRAY<(STRING OR BINARY)>\""
        );
        assert_eq!(T::Array(Box::new(T::Long)).sql(), "\"ARRAY<BIGINT>\"");
    }

    #[test]
    fn arrow_type_names_match_to_sql_type() {
        use arrow::datatypes::{Field, TimeUnit};
        let name = spark_sql_type;
        assert_eq!(
            name(&DataType::Interval(IntervalUnit::YearMonth)),
            "INTERVAL YEAR TO MONTH"
        );
        assert_eq!(
            name(&DataType::Interval(IntervalUnit::MonthDayNano)),
            "INTERVAL"
        );
        assert_eq!(
            name(&DataType::Duration(TimeUnit::Microsecond)),
            "INTERVAL DAY TO SECOND"
        );
        assert_eq!(name(&DataType::Time64(TimeUnit::Nanosecond)), "TIME(6)");
        let fields = vec![
            Field::new("a", DataType::Int32, true),
            Field::new("a b", DataType::Int32, false),
            Field::new("c`d", DataType::Utf8, true),
        ];
        assert_eq!(
            name(&DataType::Struct(fields.into())),
            "STRUCT<a: INT, `a b`: INT NOT NULL, `c``d`: STRING>"
        );
    }
}
