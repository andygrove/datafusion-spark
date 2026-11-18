// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Conversions between Spark's `TIMESTAMP` and `TIMESTAMP_NTZ` arrays.
//!
//! Spark stores a `TIMESTAMP` as microseconds since the epoch in UTC, and a
//! `TIMESTAMP_NTZ` as a local date-time, in microseconds since the epoch as if
//! it were UTC. Converting between them needs the session time zone.

use arrow::array::{PrimitiveArray, TimestampMicrosecondArray};
use arrow::datatypes::TimestampMicrosecondType;
use arrow::temporal_conversions::MICROSECONDS;
use datafusion_common::{Result, exec_datafusion_err};

use crate::SparkTimeZone;

/// Converts UTC timestamps to local date-times in `tz`, like Spark's
/// `convertTz(ts, ZoneOffset.UTC, zoneId)`. The result has no time zone.
pub(crate) fn timestamp_to_local(
    array: &TimestampMicrosecondArray,
    tz: &SparkTimeZone,
) -> Result<TimestampMicrosecondArray> {
    let rules = tz.rules();
    match rules.fixed_offset_seconds() {
        Some(seconds) => add_micros(array, i64::from(seconds) * MICROSECONDS),
        None => array
            .try_unary(|ts| rules.adjust_to_local_time::<TimestampMicrosecondType>(ts)),
    }
}

/// Converts local date-times in `tz` to UTC timestamps, the way Java's
/// `LocalDateTime.atZone` resolves daylight saving gaps and overlaps. The
/// result has no time zone.
pub(crate) fn local_to_timestamp(
    array: &TimestampMicrosecondArray,
    tz: &SparkTimeZone,
) -> Result<TimestampMicrosecondArray> {
    let rules = tz.rules();
    match rules.fixed_offset_seconds() {
        Some(seconds) => add_micros(array, -i64::from(seconds) * MICROSECONDS),
        None => array.try_unary(|ts| rules.local_to_utc::<TimestampMicrosecondType>(ts)),
    }
}

/// Adds `delta` to every value, dropping the array's time zone.
fn add_micros(
    array: &TimestampMicrosecondArray,
    delta: i64,
) -> Result<TimestampMicrosecondArray> {
    if delta == 0 {
        let (_, values, nulls) = array.clone().into_parts();
        return Ok(PrimitiveArray::new(values, nulls));
    }
    array.try_unary(|ts| {
        ts.checked_add(delta)
            .ok_or_else(|| exec_datafusion_err!("Timestamp {ts} overflows"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use chrono::NaiveDateTime;

    fn micros(datetime: &str) -> i64 {
        NaiveDateTime::parse_from_str(datetime, "%Y-%m-%d %H:%M:%S%.f")
            .unwrap()
            .and_utc()
            .timestamp_micros()
    }

    fn array(datetime: &str) -> TimestampMicrosecondArray {
        TimestampMicrosecondArray::from(vec![Some(micros(datetime)), None])
    }

    fn tz(id: &str) -> SparkTimeZone {
        SparkTimeZone::parse(id).unwrap()
    }

    #[test]
    fn test_local_to_timestamp_non_existent_time() {
        // 01:30 does not exist in London on 2024-03-31. Spark moves it forward an
        // hour, to 02:30 BST, which is 01:30 UTC.
        let result =
            local_to_timestamp(&array("2024-03-31 01:30:00"), &tz("Europe/London"))
                .unwrap();
        assert_eq!(result.value(0), micros("2024-03-31 01:30:00"));
        assert!(result.is_null(1));
    }

    #[test]
    fn test_local_to_timestamp_ambiguous_time() {
        // 01:30 happens twice in London on 2024-10-27. Spark picks the earlier
        // offset, BST, which is 00:30 UTC.
        let result =
            local_to_timestamp(&array("2024-10-27 01:30:00"), &tz("Europe/London"))
                .unwrap();
        assert_eq!(result.value(0), micros("2024-10-27 00:30:00"));
    }

    #[test]
    fn test_local_to_timestamp_keeps_fraction() {
        let result = local_to_timestamp(
            &array("2024-01-15 10:30:00.123456"),
            &tz("America/New_York"),
        )
        .unwrap();
        assert_eq!(result.value(0), micros("2024-01-15 15:30:00.123456"));
    }

    #[test]
    fn test_system_v_daylight_saving() {
        // SystemV/PST8PDT is UTC-7 in summer and UTC-8 in winter.
        let pst8pdt = tz("SystemV/PST8PDT");
        let summer = local_to_timestamp(&array("2024-07-04 12:00:00"), &pst8pdt).unwrap();
        assert_eq!(summer.value(0), micros("2024-07-04 19:00:00"));
        let winter = local_to_timestamp(&array("2024-01-15 12:00:00"), &pst8pdt).unwrap();
        assert_eq!(winter.value(0), micros("2024-01-15 20:00:00"));
        // 01:30 happens twice on 2024-10-27, and resolves to daylight time.
        let overlap =
            local_to_timestamp(&array("2024-10-27 01:30:00"), &pst8pdt).unwrap();
        assert_eq!(overlap.value(0), micros("2024-10-27 08:30:00"));
        // 02:30 does not exist on 2024-04-28, and moves forward to 03:30 PDT.
        let gap = local_to_timestamp(&array("2024-04-28 02:30:00"), &pst8pdt).unwrap();
        assert_eq!(gap.value(0), micros("2024-04-28 10:30:00"));
    }

    #[test]
    fn test_timestamp_to_local() {
        // New York is UTC-5 in January and UTC-4 in July.
        let new_york = tz("America/New_York");
        let winter =
            timestamp_to_local(&array("2024-01-15 15:30:00"), &new_york).unwrap();
        assert_eq!(winter.value(0), micros("2024-01-15 10:30:00"));
        assert!(winter.is_null(1));
        let summer =
            timestamp_to_local(&array("2024-07-04 16:30:00"), &new_york).unwrap();
        assert_eq!(summer.value(0), micros("2024-07-04 12:30:00"));
        // Java's short IDs are accepted, as in Spark.
        let pst = timestamp_to_local(&array("2024-01-15 20:00:00"), &tz("PST")).unwrap();
        assert_eq!(pst.value(0), micros("2024-01-15 12:00:00"));
    }

    #[test]
    fn test_fixed_offsets() {
        let input = array("2024-01-15 15:30:00");
        let utc = timestamp_to_local(&input, &tz("UTC")).unwrap();
        assert_eq!(utc.values(), input.values());
        assert_eq!(utc.nulls(), input.nulls());

        let plus_5_30 = tz("+05:30");
        let local = timestamp_to_local(&input, &plus_5_30).unwrap();
        assert_eq!(local.value(0), micros("2024-01-15 21:00:00"));
        let back = local_to_timestamp(&local, &plus_5_30).unwrap();
        assert_eq!(back.value(0), input.value(0));
    }

    #[test]
    fn test_out_of_range_timestamp_is_an_error() {
        let input = TimestampMicrosecondArray::from(vec![i64::MAX]);
        assert!(timestamp_to_local(&input, &tz("Europe/Copenhagen")).is_err());
        assert!(timestamp_to_local(&input, &tz("+01:00")).is_err());
    }
}
