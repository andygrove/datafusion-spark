// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared code for the date functions: the argument coercions, and the
//! [`DatePart`] to name mapping that `dayname` and `monthname` use.

use std::sync::Arc;

use arrow::array::{AsArray, StringArray};
use arrow::compute::{DatePart, date_part};
use arrow::datatypes::Int32Type;
use datafusion_common::types::{NativeType, logical_date, logical_int32, logical_string};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    Coercion, ColumnarValue, Signature, TypeSignatureClass, Volatility,
};

/// One date argument. Strings and timestamps are implicitly cast to dates, as
/// Spark does. The string cast uses DataFusion's cast, which errors on a
/// malformed string regardless of ANSI mode. A timestamp with a time zone is
/// converted to a date in its own time zone, not the session time zone.
pub(super) fn date_signature() -> Signature {
    Signature::coercible(
        vec![Coercion::new_implicit(
            TypeSignatureClass::Native(logical_date()),
            vec![
                TypeSignatureClass::Native(logical_string()),
                TypeSignatureClass::Timestamp,
            ],
            NativeType::Date,
        )],
        Volatility::Immutable,
    )
}

/// One INT argument. Spark implicitly casts other numeric and string types to
/// INT, and DataFusion types bare integer literals as BIGINT. The cast is
/// DataFusion's, which errors in cases where Spark with ANSI mode off returns a
/// value or NULL (see <https://github.com/apache/datafusion-spark/issues/88>). A
/// stopgap until input types are declared and coerced in one place
/// (<https://github.com/apache/datafusion-spark/issues/346>).
pub(super) fn int32_coercion() -> Coercion {
    Coercion::new_implicit(
        TypeSignatureClass::Native(logical_int32()),
        vec![
            TypeSignatureClass::Integer,
            TypeSignatureClass::Float,
            TypeSignatureClass::Decimal,
            TypeSignatureClass::Native(logical_string()),
        ],
        NativeType::Int32,
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
