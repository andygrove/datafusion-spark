// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, GenericStringArray};

/// Removes trailing zeros from the fraction of a second in timestamps that
/// Arrow formatted, as Spark does:
///
/// | Arrow                     | Spark                    |
/// |---------------------------|--------------------------|
/// | `1970-01-01 05:29:59.900` | `1970-01-01 05:29:59.9`  |
/// | `1970-01-01 05:29:59.990` | `1970-01-01 05:29:59.99` |
/// | `1970-01-01 05:30:00`     | `1970-01-01 05:30:00`    |
/// | `1970-01-01 05:30:00.001` | `1970-01-01 05:30:00.001`|
pub(super) fn remove_trailing_zeroes(array: &ArrayRef) -> ArrayRef {
    let result = array
        .as_string::<i32>()
        .iter()
        .map(|s| s.map(trim_fraction_zeroes))
        .collect::<GenericStringArray<i32>>();
    Arc::new(result)
}

fn trim_fraction_zeroes(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;

    #[test]
    fn test_drops_trailing_zeroes() {
        let strings: ArrayRef = Arc::new(StringArray::from(vec![
            Some("1970-01-01 05:29:59.900"),
            Some("1970-01-01 05:29:59.990"),
            Some("1970-01-01 05:29:59.999"),
            Some("1970-01-01 05:30:00"),
            Some("1970-01-01 05:30:00.001"),
            Some("1970-01-01 05:30:00.123400"),
            None,
        ]));
        assert_eq!(
            remove_trailing_zeroes(&strings).as_string::<i32>(),
            &StringArray::from(vec![
                Some("1970-01-01 05:29:59.9"),
                Some("1970-01-01 05:29:59.99"),
                Some("1970-01-01 05:29:59.999"),
                Some("1970-01-01 05:30:00"),
                Some("1970-01-01 05:30:00.001"),
                Some("1970-01-01 05:30:00.1234"),
                None,
            ])
        );
    }
}
