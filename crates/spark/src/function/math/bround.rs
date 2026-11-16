// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::Write;
use std::sync::Arc;

use arrow::array::*;
use arrow::compute::CastOptions;
use arrow::datatypes::{
    DECIMAL128_MAX_PRECISION, DataType, Decimal128Type, DecimalType, Field, FieldRef,
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, exec_err, internal_err, plan_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::error_utils::ARITHMETIC_OVERFLOW_ERROR;

/// Spark-compatible `bround` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#bround>
#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Rounds `expr` to `d` decimal places using HALF_EVEN rounding: a tie goes to the even \
        neighbour (2.5 becomes 2, 3.5 becomes 4). A negative `d` rounds to the left of the decimal point. \
        Integers and floating-point values keep their type, and a string is cast to DOUBLE (a malformed \
        string is NULL with `datafusion.execution.enable_ansi_mode` off and an error with it on). \
        A decimal `DECIMAL(p, s)` returns `DECIMAL(min(p - s + 1 + min(s, d), 38), min(s, d))` for `d >= 0` \
        and `DECIMAL(min(max(p - s + 1, 1 - d), 38), 0)` for `d < 0`, like Spark. Floating-point values are \
        rounded from their shortest decimal representation, so `bround(2.675, 2)` is 2.68. Integer overflow \
        wraps with ANSI mode off and fails with it on. Infinity and NaN are returned unchanged. \
        Returns NULL if `expr` or `d` is NULL.",
    syntax_example = "bround(expr[, d])",
    sql_example = r#"```sql
> SELECT bround(2.5, 0);
2
> SELECT bround(25, -1);
20
```"#,
    argument(
        name = "expr",
        description = "Numeric or string expression. NULL returns NULL."
    ),
    argument(
        name = "d",
        description = "Optional constant integer number of decimal places. Defaults to 0. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkBRound {
    signature: Signature,
}

impl Default for SparkBRound {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkBRound {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkBRound {
    fn name(&self) -> &str {
        "bround"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let (value_type, scale_type) = match arg_types {
            [value_type] => (value_type, None),
            [value_type, scale_type] => (value_type, Some(scale_type)),
            _ => {
                return plan_err!(
                    "bround requires 1 or 2 arguments, got {}",
                    arg_types.len()
                );
            }
        };
        // A dictionary is coerced to its value type
        let value_type = match value_type {
            DataType::Dictionary(_, value_type) => value_type.as_ref(),
            other => other,
        };
        let value_type = match value_type {
            // A NULL literal is treated as DOUBLE, like Spark's implicit cast
            DataType::Null => DataType::Float64,
            t if t.is_signed_integer()
                || t.is_string()
                || matches!(t, DataType::Float32 | DataType::Float64) =>
            {
                t.clone()
            }
            DataType::Decimal32(p, s)
            | DataType::Decimal64(p, s)
            | DataType::Decimal128(p, s)
            | DataType::Decimal256(p, s) => {
                if *p > DECIMAL128_MAX_PRECISION {
                    return plan_err!(
                        "bround supports decimals up to precision 38, got {value_type}"
                    );
                }
                DataType::Decimal128(*p, *s)
            }
            other => {
                return plan_err!(
                    "bround requires a numeric or string input, got {other}"
                );
            }
        };
        let mut coerced = vec![value_type];
        if let Some(scale_type) = scale_type {
            if !(scale_type.is_null() || scale_type.is_signed_integer()) {
                return plan_err!("bround scale must be an integer, got {scale_type}");
            }
            coerced.push(DataType::Int32);
        }
        Ok(coerced)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let return_type = match args.arg_fields[0].data_type() {
            t if t.is_string() => DataType::Float64,
            DataType::Decimal128(precision, input_scale) => {
                let scale = match args.scalar_arguments.get(1) {
                    None => 0,
                    Some(Some(scalar)) => scale_from_scalar(scalar)?.unwrap_or(0),
                    // A constant scale that is not a literal yet (e.g. `1::bigint`) is
                    // folded to one by the optimizer, which then computes the final type
                    Some(None) => i32::from(*input_scale),
                };
                decimal_return_type(*precision, *input_scale, scale)
            }
            other => other.clone(),
        };
        Ok(Arc::new(Field::new(self.name(), return_type, true)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let enable_ansi_mode = args.config_options.execution.enable_ansi_mode;
        let return_type = args.return_field.data_type();
        let scale = match args.args.get(1) {
            None => Some(0),
            Some(ColumnarValue::Scalar(scalar)) => scale_from_scalar(scalar)?,
            Some(other) => {
                return exec_err!(
                    "bround requires a constant scale, got {}",
                    other.data_type()
                );
            }
        };
        let Some(scale) = scale else {
            return Ok(ColumnarValue::Scalar(ScalarValue::try_from(return_type)?));
        };

        let mut value = args.args[0].clone();
        if value.data_type().is_string() {
            // A malformed string is NULL in legacy mode and an error under ANSI
            let options = CastOptions {
                safe: !enable_ansi_mode,
                ..Default::default()
            };
            value = value.cast_to(&DataType::Float64, Some(&options))?;
        }
        let is_scalar = matches!(value, ColumnarValue::Scalar(_));
        let array = value.to_array(1)?;
        let result = bround_array(&array, scale, return_type, enable_ansi_mode)?;
        if is_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

/// Returns the scale, or `None` if it is NULL
fn scale_from_scalar(scalar: &ScalarValue) -> Result<Option<i32>> {
    if scalar.is_null() {
        return Ok(None);
    }
    match scalar.cast_to(&DataType::Int32)? {
        ScalarValue::Int32(Some(v)) => Ok(Some(v)),
        _ => exec_err!("Unsupported type for bround scale: {}", scalar.data_type()),
    }
}

/// Spark's `RoundBase` result type for `DECIMAL(p, s)`
fn decimal_return_type(precision: u8, input_scale: i8, scale: i32) -> DataType {
    let (p, s) = (i32::from(precision), i32::from(input_scale));
    let max = i32::from(DECIMAL128_MAX_PRECISION);
    let (new_p, new_s) = if scale >= 0 {
        let new_s = s.min(scale);
        ((p - s + 1 + new_s).min(max), new_s)
    } else {
        ((p - s + 1).max(1 - scale).min(max), 0)
    };
    DataType::Decimal128(new_p as u8, new_s as i8)
}

/// HALF_EVEN rounding of an integer to `scale` digits (only negative `scale` changes it)
fn bround_integer(value: i128, scale: i32) -> i128 {
    if scale >= 0 {
        return value;
    }
    let Some(factor) = 10_i128.checked_pow(scale.unsigned_abs()) else {
        return 0;
    };
    let (quotient, remainder) = (value / factor, value % factor);
    let half = factor / 2;
    let away = remainder.abs() > half || (remainder.abs() == half && quotient % 2 != 0);
    match (away, remainder > 0) {
        (false, _) => value - remainder,
        (true, true) => value - remainder + factor,
        (true, false) => value - remainder - factor,
    }
}

/// HALF_EVEN rounding of an unscaled decimal to `scale` digits, returned at `new_scale`
fn bround_decimal(value: i128, input_scale: i8, scale: i32, new_scale: i8) -> i128 {
    let dropped = i64::from(input_scale) - i64::from(scale);
    if dropped <= 0 {
        return value;
    }
    // `rounded` is still at `input_scale`, with the dropped digits set to zero
    let rounded = match i32::try_from(dropped) {
        Ok(dropped) => bround_integer(value, -dropped),
        Err(_) => 0,
    };
    let shift = u32::try_from(i32::from(input_scale) - i32::from(new_scale)).unwrap_or(0);
    10_i128
        .checked_pow(shift)
        .map_or(0, |factor| rounded / factor)
}

/// A small stack buffer, to format floats without allocating
struct StackBuffer {
    bytes: [u8; 40],
    len: usize,
}

impl StackBuffer {
    fn new() -> Self {
        Self {
            bytes: [0; 40],
            len: 0,
        }
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

impl Write for StackBuffer {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let end = self.len + s.len();
        if end > self.bytes.len() {
            return Err(std::fmt::Error);
        }
        self.bytes[self.len..end].copy_from_slice(s.as_bytes());
        self.len = end;
        Ok(())
    }
}

/// A double rounded HALF_EVEN, like Spark's `BigDecimal(d).setScale(scale, HALF_EVEN)`
enum RoundedFloat {
    /// No digit is dropped, so the value is unchanged
    Unchanged,
    /// The rounded magnitude is `digits / 10^scale`
    Digits(u64, i32),
}

/// Rounds the shortest decimal representation of a finite, non-zero `value`
fn bround_float_digits(value: f64, scale: i32) -> RoundedFloat {
    let scale64 = i64::from(scale);
    // The shortest representation has at most 17 significant digits, and the
    // estimated decimal exponent can be off by one
    let exponent_estimate = value.abs().log10().floor() as i64;
    if scale64 >= 17 - exponent_estimate {
        return RoundedFloat::Unchanged;
    }
    if -scale64 >= exponent_estimate + 3 {
        return RoundedFloat::Digits(0, 0);
    }

    let mut buffer = StackBuffer::new();
    if write!(buffer, "{:e}", value.abs()).is_err() {
        return RoundedFloat::Unchanged;
    }
    let (mantissa, exponent) = buffer.as_str().split_once('e').unwrap_or(("0", "0"));
    let mut digits: u64 = 0;
    let mut fraction_len: i64 = 0;
    let mut in_fraction = false;
    for byte in mantissa.bytes() {
        if byte == b'.' {
            in_fraction = true;
        } else {
            digits = digits * 10 + u64::from(byte - b'0');
            fraction_len += i64::from(in_fraction);
        }
    }
    let last_digit_exponent = exponent.parse::<i64>().unwrap_or(0) - fraction_len;

    let dropped = -scale64 - last_digit_exponent;
    if dropped <= 0 {
        return RoundedFloat::Unchanged;
    }
    let Some(factor) = u32::try_from(dropped)
        .ok()
        .and_then(|dropped| 10_u64.checked_pow(dropped))
    else {
        return RoundedFloat::Digits(0, 0);
    };
    let (quotient, remainder) = (digits / factor, digits % factor);
    let half = factor / 2;
    let rounded = if remainder > half || (remainder == half && quotient % 2 == 1) {
        quotient + 1
    } else {
        quotient
    };
    RoundedFloat::Digits(rounded, scale)
}

/// Parses `digits / 10^scale` with a single rounding
fn parse_digits<T: std::str::FromStr + Default>(digits: u64, scale: i32) -> T {
    let mut buffer = StackBuffer::new();
    match write!(buffer, "{digits}e{}", -i64::from(scale)) {
        Ok(()) => buffer.as_str().parse().unwrap_or_default(),
        Err(_) => T::default(),
    }
}

const F64_POWERS_OF_TEN: [f64; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

const F32_POWERS_OF_TEN: [f32; 11] =
    [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10];

fn bround_f64(value: f64, scale: i32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        // BigDecimal has no negative zero
        return 0.0;
    }
    let (digits, scale) = match bround_float_digits(value, scale) {
        RoundedFloat::Unchanged => return value,
        RoundedFloat::Digits(0, _) => return 0.0,
        RoundedFloat::Digits(digits, scale) => (digits, scale),
    };
    let power = scale.unsigned_abs() as usize;
    // Both operands are exact, so one operation rounds correctly
    let magnitude = if digits < (1 << 53) && power < F64_POWERS_OF_TEN.len() {
        if scale >= 0 {
            digits as f64 / F64_POWERS_OF_TEN[power]
        } else {
            digits as f64 * F64_POWERS_OF_TEN[power]
        }
    } else {
        parse_digits(digits, scale)
    };
    if value < 0.0 { -magnitude } else { magnitude }
}

/// Spark widens FLOAT to DOUBLE, rounds, and converts the decimal straight to FLOAT
fn bround_f32(value: f32, scale: i32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return 0.0;
    }
    let (digits, scale) = match bround_float_digits(f64::from(value), scale) {
        RoundedFloat::Unchanged => return value,
        RoundedFloat::Digits(0, _) => return 0.0,
        RoundedFloat::Digits(digits, scale) => (digits, scale),
    };
    let power = scale.unsigned_abs() as usize;
    let magnitude = if digits < (1 << 24) && power < F32_POWERS_OF_TEN.len() {
        if scale >= 0 {
            digits as f32 / F32_POWERS_OF_TEN[power]
        } else {
            digits as f32 * F32_POWERS_OF_TEN[power]
        }
    } else {
        parse_digits(digits, scale)
    };
    if value < 0.0 { -magnitude } else { magnitude }
}

macro_rules! bround_integer_array {
    ($array:expr, $arrow_type:ty, $scale:expr, $enable_ansi_mode:expr) => {{
        type Native = <$arrow_type as arrow::datatypes::ArrowPrimitiveType>::Native;
        let array = $array.as_primitive::<$arrow_type>();
        let result: PrimitiveArray<$arrow_type> = if $enable_ansi_mode {
            array.try_unary(|x| {
                Native::try_from(bround_integer(i128::from(x), $scale))
                    .map_err(|_| exec_datafusion_err!("{ARITHMETIC_OVERFLOW_ERROR}"))
            })?
        } else {
            array.unary(|x| bround_integer(i128::from(x), $scale) as Native)
        };
        Arc::new(result) as ArrayRef
    }};
}

fn bround_array(
    array: &ArrayRef,
    scale: i32,
    return_type: &DataType,
    enable_ansi_mode: bool,
) -> Result<ArrayRef> {
    Ok(match (array.data_type(), return_type) {
        (DataType::Int8, _) => {
            bround_integer_array!(array, Int8Type, scale, enable_ansi_mode)
        }
        (DataType::Int16, _) => {
            bround_integer_array!(array, Int16Type, scale, enable_ansi_mode)
        }
        (DataType::Int32, _) => {
            bround_integer_array!(array, Int32Type, scale, enable_ansi_mode)
        }
        (DataType::Int64, _) => {
            bround_integer_array!(array, Int64Type, scale, enable_ansi_mode)
        }
        (DataType::Float32, _) => Arc::new(
            array
                .as_primitive::<Float32Type>()
                .unary::<_, Float32Type>(|x| bround_f32(x, scale)),
        ),
        (DataType::Float64, _) => Arc::new(
            array
                .as_primitive::<Float64Type>()
                .unary::<_, Float64Type>(|x| bround_f64(x, scale)),
        ),
        (
            DataType::Decimal128(_, input_scale),
            DataType::Decimal128(precision, new_scale),
        ) => {
            let result = array
                .as_primitive::<Decimal128Type>()
                .try_unary::<_, Decimal128Type, _>(|x| {
                    let rounded = bround_decimal(x, *input_scale, scale, *new_scale);
                    Decimal128Type::validate_decimal_precision(
                        rounded, *precision, *new_scale,
                    )
                    .map_err(|_| {
                        exec_datafusion_err!(
                            "[NUMERIC_VALUE_OUT_OF_RANGE] The rounded value cannot be represented as Decimal({precision}, {new_scale})"
                        )
                    })?;
                    Ok::<_, datafusion_common::DataFusionError>(rounded)
                })?;
            Arc::new(result.with_data_type(return_type.clone()))
        }
        (dt, _) => return exec_err!("Unsupported data type for Spark bround(): {dt}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The arithmetic fast path must match parsing the rounded decimal
    #[test]
    fn test_float_fast_path_matches_parsing() {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut checked = 0;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            let value = (unit - 0.5) * 10f64.powi((state % 24) as i32 - 6);
            if !value.is_finite() || value == 0.0 {
                continue;
            }
            let scale = ((state >> 5) % 21) as i32 - 6;
            if let RoundedFloat::Digits(digits, scale) = bround_float_digits(value, scale)
                && digits != 0
            {
                checked += 1;
                let parsed: f64 = parse_digits(digits, scale);
                assert_eq!(bround_f64(value, scale).abs(), parsed, "{value} {scale}");
                let value32 = value as f32;
                if value32.is_finite()
                    && value32 != 0.0
                    && let RoundedFloat::Digits(digits, scale) =
                        bround_float_digits(f64::from(value32), scale)
                    && digits != 0
                {
                    let parsed: f32 = parse_digits(digits, scale);
                    assert_eq!(bround_f32(value32, scale).abs(), parsed);
                }
            }
        }
        assert!(checked > 100_000, "only {checked} values were rounded");
        for (value, scale, expected) in [
            (2.5, 0, 2.0),
            (2.675, 2, 2.68),
            (0.125, 2, 0.12),
            (-0.35, 1, -0.4),
            (1234.5678, 2, 1234.57),
            (1.5e300, -300, 2e300),
        ] {
            assert_eq!(bround_f64(value, scale), expected);
        }
    }
}
