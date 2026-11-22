// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark's rules for finding a common type of several arguments, from its `TypeCoercion`
//! and `AnsiTypeCoercion`, on the Arrow types that Spark's types map to. Also the checks
//! and type names that the errors for those arguments use.
//!
//! These are for functions that need one type for several arguments, such as
//! `array_position`. They are not Spark's implicit casts to a function's declared input
//! types, which the functions do through their [`datafusion_expr::Signature`].

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, IntervalUnit};

/// Spark's `findTightestCommonType`. `ansi_mode` picks between `AnsiTypeCoercion` and
/// `TypeCoercion`, which differ only in that ANSI widens an integer and a FLOAT to DOUBLE.
/// Arrays, maps and structs are widened element by element with this function.
pub(crate) fn tightest_common_type(
    left: &DataType,
    right: &DataType,
    ansi_mode: bool,
) -> Option<DataType> {
    use DataType::*;
    // Spark has no dictionary types. A dictionary-encoded argument, such as a
    // Parquet string column, has the type of its values.
    if let Dictionary(_, value) = left {
        return tightest_common_type(value, right, ansi_mode);
    }
    if let Dictionary(_, value) = right {
        return tightest_common_type(left, value, ansi_mode);
    }
    // Nested types are widened field by field below, which also unwraps nested dictionaries.
    if left == right && !left.is_nested() {
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
        (l, r) if l.is_string() && r.is_string() => Some(wider_physical_type(l, r)),
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
            // Arrow orders the units from Second to Nanosecond.
            (*lu).max(*ru),
            ltz.clone().or_else(|| rtz.clone()),
        )),
        _ => complex_type(left, right, &|l, r| tightest_common_type(l, r, ansi_mode)),
    }
}

/// Spark's `findWiderTypeWithoutStringPromotionForTwo`: the tightest common type, then a
/// decimal wide enough for both, then the same for the elements of arrays and the fields of
/// structs. Strings are never promoted. `ansi_mode` picks between `AnsiTypeCoercion` and
/// `TypeCoercion`, which differ only in that ANSI widens an integer and a FLOAT to DOUBLE.
pub(crate) fn wider_type(
    left: &DataType,
    right: &DataType,
    ansi_mode: bool,
) -> Option<DataType> {
    tightest_common_type(left, right, ansi_mode)
        .or_else(|| wider_decimal_type(left, right))
        .or_else(|| complex_type(left, right, &|l, r| wider_type(l, r, ansi_mode)))
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

/// Spark's `findTypeForComplex`: widens the elements of arrays, the keys and values of maps,
/// and the fields of structs with `find_type`. A nested field is nullable if it is on either
/// side. Struct fields must have the same names in the same order. Spark compares them
/// ignoring case, but DataFusion cannot cast between structs whose field names differ only
/// in case, so this requires an exact match.
fn complex_type(
    left: &DataType,
    right: &DataType,
    find_type: &dyn Fn(&DataType, &DataType) -> Option<DataType>,
) -> Option<DataType> {
    use DataType::*;
    match (left, right) {
        (
            List(lf) | LargeList(lf) | FixedSizeList(lf, _),
            List(rf) | LargeList(rf) | FixedSizeList(rf, _),
        ) => {
            let item = find_type(lf.data_type(), rf.data_type())?;
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
                    let data_type = find_type(a.data_type(), b.data_type())?;
                    Some(Field::new(
                        a.name(),
                        data_type,
                        a.is_nullable() || b.is_nullable(),
                    ))
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Struct(Fields::from(fields)))
        }
        // The entries are a struct of the key and the value. Spark's `MapType` has no field
        // names, so the key and the value are matched by position.
        (Map(lf, sorted), Map(rf, _)) => {
            let (Struct(lkv), Struct(rkv)) = (lf.data_type(), rf.data_type()) else {
                return None;
            };
            let ([lk, lv], [rk, rv]) = (&lkv[..], &rkv[..]) else {
                return None;
            };
            // Spark rejects a key type that the cast can make NULL (`Cast.forceNullable`).
            let key = find_type(lk.data_type(), rk.data_type()).filter(|key| {
                !force_nullable(lk.data_type(), key)
                    && !force_nullable(rk.data_type(), key)
            })?;
            let value = find_type(lv.data_type(), rv.data_type())?;
            let entries = Struct(Fields::from(vec![
                Field::new(lk.name(), key, false),
                Field::new(lv.name(), value, lv.is_nullable() || rv.is_nullable()),
            ]));
            Some(Map(
                Arc::new(Field::new(lf.name(), entries, false)),
                *sorted,
            ))
        }
        _ => None,
    }
}

/// Spark's `Cast.forceNullable`, for the casts that these rules produce. Of those, Spark
/// marks only DATE to TIMESTAMP_NTZ as one that can return NULL.
fn force_nullable(from: &DataType, to: &DataType) -> bool {
    match from {
        DataType::Dictionary(_, value) => force_nullable(value, to),
        _ => matches!((from, to), (DataType::Date32, DataType::Timestamp(_, None))),
    }
}

/// Spark's `OrderUtils.isOrderable`. Spark cannot order maps or `CalendarIntervalType`, which
/// is `Interval(MonthDayNano)` in Arrow, at any depth.
pub(crate) fn is_orderable(data_type: &DataType) -> bool {
    match data_type {
        DataType::Map(_, _)
        | DataType::Union(_, _)
        | DataType::Interval(IntervalUnit::MonthDayNano) => false,
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

/// Whether the structs in two types, at any depth, have the same field types. With ANSI mode
/// off, Spark's implicit cast cannot change the field types of a struct, so a function that
/// casts its arguments to their tightest common type rejects such structs, as
/// `array_position` does. With ANSI mode on, Spark widens the field types.
pub(crate) fn struct_field_types_match(left: &DataType, right: &DataType) -> bool {
    use DataType::*;
    match (left, right) {
        (Dictionary(_, l), r) | (r, Dictionary(_, l)) => struct_field_types_match(l, r),
        (
            List(l) | LargeList(l) | FixedSizeList(l, _),
            List(r) | LargeList(r) | FixedSizeList(r, _),
        ) => struct_field_types_match(l.data_type(), r.data_type()),
        // The entries are a struct of the key and the value, which are checked on their own.
        (Map(l, _), Map(r, _)) => match (l.data_type(), r.data_type()) {
            (Struct(l), Struct(r)) => l
                .iter()
                .zip(r.iter())
                .all(|(a, b)| struct_field_types_match(a.data_type(), b.data_type())),
            _ => true,
        },
        (Struct(_), Struct(_)) => same_type(left, right),
        _ => true,
    }
}

/// Whether two types are equal when the nullability of nested fields is ignored, like
/// Spark's `DataTypeUtils.sameType`.
fn same_type(left: &DataType, right: &DataType) -> bool {
    use DataType::*;
    match (left, right) {
        (Dictionary(_, l), r) | (r, Dictionary(_, l)) => same_type(l, r),
        (List(l), List(r)) | (LargeList(l), LargeList(r)) | (Map(l, _), Map(r, _)) => {
            same_type(l.data_type(), r.data_type())
        }
        (FixedSizeList(l, ln), FixedSizeList(r, rn)) => {
            ln == rn && same_type(l.data_type(), r.data_type())
        }
        (Struct(l), Struct(r)) => {
            l.len() == r.len()
                && l.iter().zip(r.iter()).all(|(a, b)| {
                    a.name() == b.name() && same_type(a.data_type(), b.data_type())
                })
        }
        _ => left == right,
    }
}

/// The name Spark's `toSQLType` gives the Spark type that an Arrow type maps to, for error
/// messages. Arrow types without a Spark counterpart use their Arrow name.
pub(crate) fn spark_sql_type(data_type: &DataType) -> String {
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

/// Unlike [`DataType::is_binary`], this leaves out `FixedSizeBinary`, which
/// [`wider_physical_type`] cannot widen.
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

#[cfg(test)]
mod tests {
    use super::*;
    use DataType::*;

    fn list(item: DataType) -> DataType {
        List(Arc::new(Field::new("item", item, true)))
    }

    fn map(key: DataType, value: DataType) -> DataType {
        let entries = Struct(Fields::from(vec![
            Field::new("key", key, false),
            Field::new("value", value, true),
        ]));
        Map(Arc::new(Field::new("entries", entries, false)), false)
    }

    #[test]
    fn numbers_widen_and_ansi_widens_int_and_float_to_double() {
        assert_eq!(tightest_common_type(&Int32, &Int64, false), Some(Int64));
        assert_eq!(tightest_common_type(&Int32, &Float32, false), Some(Float32));
        assert_eq!(tightest_common_type(&Int32, &Float32, true), Some(Float64));
        assert_eq!(
            tightest_common_type(&Int32, &Decimal128(10, 0), false),
            Some(Decimal128(10, 0))
        );
        assert_eq!(
            tightest_common_type(&Int32, &Decimal128(10, 1), false),
            None
        );
        assert_eq!(tightest_common_type(&Int32, &Utf8, false), None);
    }

    #[test]
    fn string_layouts_widen_to_the_one_that_holds_both() {
        assert_eq!(
            tightest_common_type(&Utf8, &LargeUtf8, false),
            Some(LargeUtf8)
        );
        assert_eq!(
            tightest_common_type(&LargeUtf8, &Utf8, false),
            Some(LargeUtf8)
        );
        assert_eq!(
            tightest_common_type(&Utf8, &Utf8View, false),
            Some(Utf8View)
        );
    }

    #[test]
    fn complex_types_widen_element_by_element() {
        assert_eq!(
            tightest_common_type(&list(Int32), &list(Int64), false),
            Some(list(Int64))
        );
        assert_eq!(
            tightest_common_type(&map(Int32, Int32), &map(Int64, Int16), false),
            Some(map(Int64, Int32))
        );
        let a = Struct(Fields::from(vec![Field::new("a", Int32, false)]));
        let b = Struct(Fields::from(vec![Field::new("b", Int32, false)]));
        // Map entries named differently are matched by position.
        let renamed = Struct(Fields::from(vec![
            Field::new("keys", Int64, false),
            Field::new("values", Int16, true),
        ]));
        let renamed = Map(Arc::new(Field::new("entries", renamed, false)), false);
        assert_eq!(
            tightest_common_type(&map(Int32, Int32), &renamed, false),
            Some(map(Int64, Int32))
        );
        let wide = Struct(Fields::from(vec![Field::new("a", Int64, true)]));
        assert_eq!(tightest_common_type(&a, &wide, false), Some(wide));
        assert_eq!(tightest_common_type(&a, &b, false), None);
    }

    #[test]
    fn map_keys_do_not_widen_from_date_to_timestamp_ntz() {
        let ntz = Timestamp(arrow::datatypes::TimeUnit::Microsecond, None);
        assert_eq!(
            tightest_common_type(&map(Date32, Int32), &map(ntz.clone(), Int32), false),
            None
        );
        assert_eq!(
            tightest_common_type(&map(ntz.clone(), Int32), &map(Date32, Int32), true),
            None
        );
        assert_eq!(
            tightest_common_type(&map(Int32, Date32), &map(Int32, ntz.clone()), false),
            Some(map(Int32, ntz.clone()))
        );
        let date_dictionary = Dictionary(Box::new(Int32), Box::new(Date32));
        assert_eq!(
            tightest_common_type(
                &map(date_dictionary, Int32),
                &map(ntz.clone(), Int32),
                false
            ),
            None
        );
        let tz = Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()));
        assert_eq!(
            tightest_common_type(&map(Date32, Int32), &map(tz.clone(), Int32), false),
            Some(map(tz, Int32))
        );
    }

    #[test]
    fn struct_field_types_must_match_at_any_depth() {
        let int = Struct(Fields::from(vec![Field::new("a", Int32, false)]));
        let nullable_int = Struct(Fields::from(vec![Field::new("a", Int32, true)]));
        let long = Struct(Fields::from(vec![Field::new("a", Int64, false)]));
        assert!(struct_field_types_match(&int, &nullable_int));
        assert!(!struct_field_types_match(&int, &long));
        assert!(!struct_field_types_match(
            &list(int.clone()),
            &list(long.clone())
        ));
        assert!(!struct_field_types_match(
            &map(Int32, int),
            &map(Int64, long)
        ));
        assert!(struct_field_types_match(&list(Int32), &list(Int64)));
    }

    #[test]
    fn dictionaries_have_the_type_of_their_values() {
        let dictionary = |value: DataType| Dictionary(Box::new(Int32), Box::new(value));
        assert_eq!(
            tightest_common_type(&dictionary(Utf8), &Utf8, false),
            Some(Utf8)
        );
        assert_eq!(
            tightest_common_type(&Int64, &dictionary(Int32), false),
            Some(Int64)
        );
        assert_eq!(tightest_common_type(&dictionary(Utf8), &Int32, false), None);
        let field = |data_type: DataType| {
            Struct(Fields::from(vec![Field::new("a", data_type, true)]))
        };
        assert!(struct_field_types_match(
            &field(dictionary(Utf8)),
            &field(Utf8)
        ));
        assert_eq!(spark_sql_type(&dictionary(Utf8)), "STRING");
        // Two dictionaries are unwrapped too, so the function sees the values' type.
        assert_eq!(
            tightest_common_type(&dictionary(Float64), &dictionary(Float64), false),
            Some(Float64)
        );
        // So are dictionaries inside two equal nested types.
        assert_eq!(
            tightest_common_type(
                &list(dictionary(Float64)),
                &list(dictionary(Float64)),
                false
            ),
            Some(list(Float64))
        );
        assert_eq!(
            tightest_common_type(
                &field(dictionary(Float64)),
                &field(dictionary(Float64)),
                false
            ),
            Some(field(Float64))
        );
    }

    #[test]
    fn spark_names_for_intervals_times_and_struct_fields() {
        use arrow::datatypes::TimeUnit;
        assert_eq!(
            spark_sql_type(&Interval(IntervalUnit::YearMonth)),
            "INTERVAL YEAR TO MONTH"
        );
        assert_eq!(
            spark_sql_type(&Duration(TimeUnit::Microsecond)),
            "INTERVAL DAY TO SECOND"
        );
        assert_eq!(spark_sql_type(&Time64(TimeUnit::Nanosecond)), "TIME(6)");
        let fields = Struct(Fields::from(vec![
            Field::new("a_1", Int32, true),
            Field::new("a b", Int32, true),
            Field::new("1a", Int32, true),
            Field::new("a`b", Int32, false),
        ]));
        assert_eq!(
            spark_sql_type(&fields),
            "STRUCT<a_1: INT, `a b`: INT, `1a`: INT, `a``b`: INT NOT NULL>"
        );
    }

    #[test]
    fn maps_and_calendar_intervals_are_not_orderable() {
        let interval = Interval(IntervalUnit::MonthDayNano);
        assert!(!is_orderable(&map(Int32, Int32)));
        assert!(!is_orderable(&interval));
        assert!(!is_orderable(&list(interval.clone())));
        assert!(is_orderable(&list(Int32)));
        assert_eq!(spark_sql_type(&list(interval)), "ARRAY<INTERVAL>");
    }
}
