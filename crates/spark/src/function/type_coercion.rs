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

use arrow::datatypes::{DataType, Field, Fields, IntervalUnit, TimeUnit};

/// Spark's `findTightestCommonType`. `ansi_mode` picks between `AnsiTypeCoercion` and
/// `TypeCoercion`, which differ only in that ANSI widens an integer and a FLOAT to DOUBLE.
/// Arrays, maps and structs are widened element by element with this function.
pub(crate) fn tightest_common_type(
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
        _ => complex_type(left, right, &|l, r| tightest_common_type(l, r, ansi_mode)),
    }
}

/// Spark's `findTypeForComplex`: widens the elements of arrays, the keys and values of maps,
/// and the fields of structs with `find_type`. A nested field is nullable if it is on either
/// side. Struct fields must have the same names in the same order. Spark compares them
/// ignoring case, but DataFusion cannot cast between structs whose field names differ only
/// in case, so this requires an exact match.
pub(crate) fn complex_type(
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
        // The entries are a struct of the key and the value, so the struct arm widens both.
        (Map(lf, sorted), Map(rf, _)) => {
            let entries = complex_type(lf.data_type(), rf.data_type(), find_type)?;
            Some(Map(
                Arc::new(Field::new(lf.name(), entries, false)),
                *sorted,
            ))
        }
        _ => None,
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
        let wide = Struct(Fields::from(vec![Field::new("a", Int64, true)]));
        assert_eq!(tightest_common_type(&a, &wide, false), Some(wide));
        assert_eq!(tightest_common_type(&a, &b, false), None);
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
