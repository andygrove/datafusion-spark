// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared code for `dayname` and `monthname`, which map one [`DatePart`] of a
//! date to a name.

use std::sync::Arc;

use arrow::array::{AsArray, StringArray};
use arrow::compute::{DatePart, date_part};
use arrow::datatypes::Int32Type;
use datafusion_common::types::{NativeType, logical_date};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    Coercion, ColumnarValue, Signature, TypeSignatureClass, Volatility,
};

/// One date argument. Timestamps are cast to dates first, like Spark's
/// implicit cast.
pub(super) fn date_signature() -> Signature {
    Signature::coercible(
        vec![Coercion::new_implicit(
            TypeSignatureClass::Native(logical_date()),
            vec![TypeSignatureClass::Timestamp],
            NativeType::Date,
        )],
        Volatility::Immutable,
    )
}

/// Extracts `part` from each date in `arg` and maps it to a name with `name`.
/// NULL dates, and values `name` returns `None` for, give NULL.
pub(super) fn date_part_name(
    arg: ColumnarValue,
    part: DatePart,
    name: fn(i32) -> Option<&'static str>,
) -> Result<ColumnarValue> {
    match arg {
        ColumnarValue::Scalar(scalar) => {
            if scalar.is_null() {
                return Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None)));
            }
            let parts = date_part(&scalar.to_array_of_size(1)?, part)?;
            let value = name(parts.as_primitive::<Int32Type>().value(0));
            Ok(ColumnarValue::Scalar(ScalarValue::Utf8(
                value.map(str::to_string),
            )))
        }
        ColumnarValue::Array(arr) => {
            let parts = date_part(&arr, part)?;
            let result: StringArray = parts
                .as_primitive::<Int32Type>()
                .iter()
                .map(|v| v.and_then(name))
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
    }
}
