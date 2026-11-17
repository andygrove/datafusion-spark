// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt::LowerExp;
use std::sync::Arc;

use arrow::array::timezone::Tz;
use arrow::array::{
    Array, ArrayAccessor, ArrayRef, AsArray, MapArray, PrimitiveArray, StringArray,
    StringBuilder,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::cast;
use arrow::datatypes::{
    ArrowPrimitiveType, ArrowTimestampType, DataType, Date32Type, Date64Type,
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DecimalType, FieldRef,
    Float32Type, Float64Type, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType,
};
use arrow::error::ArrowError;
use arrow::json::writer::{
    Encoder, EncoderFactory, EncoderOptions, NullableEncoder, make_encoder,
};
use arrow::temporal_conversions::{
    MICROSECONDS, MILLISECONDS, MILLISECONDS_IN_DAY, NANOSECONDS, as_date, as_datetime,
    as_datetime_with_timezone,
};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use datafusion_common::config::ConfigOptions;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, ScalarValue, exec_datafusion_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion_macros::user_doc;

/// Spark-compatible `to_json` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#to_json>
#[user_doc(
    doc_section(label = "JSON Functions"),
    description = "Serializes a struct to a JSON string, one object per row, as Spark's `to_json` does with its \
        default options. DataFusion has no built-in `to_json`. \
        A NULL struct returns NULL. NULL struct fields are omitted (`ignoreNullFields`), while NULL map entries \
        are kept as `\"key\":null`. Nested structs, lists and maps are serialized recursively. \
        Floats are written like Java's `Double.toString` (`1.0`, `1.0E-5`; `NaN` and `Infinity` as strings), \
        decimals like `BigDecimal.toString` (`1.50`, `0E-10`), binary as Base64, dates as `yyyy-MM-dd`, \
        timestamps as `yyyy-MM-dd'T'HH:mm:ss.SSS` with the session time zone offset \
        (`datafusion.execution.time_zone`, `Z` for UTC) and without offset for a timestamp without time zone. \
        Map keys are written as strings holding Spark's internal value: dates as days since the epoch, \
        timestamps as microseconds, numbers as above. \
        Types with no Spark JSON representation (intervals, durations, times, unions, map keys other than \
        strings, booleans, numbers, dates and timestamps) are rejected when the query is planned. \
        The options argument and map or array input are not supported.",
    syntax_example = "to_json(struct_expr)",
    sql_example = r#"```sql
> SELECT to_json(named_struct('a', 1, 'b', 'x', 'c', CAST(NULL AS INT)));
{"a":1,"b":"x"}
> SELECT to_json(named_struct('t', TIMESTAMP '2024-01-01 01:02:03.123456', 'd', 1e-5));
{"t":"2024-01-01T01:02:03.123","d":1.0E-5}
> SELECT to_json(named_struct('m', map([DATE '2024-01-01'], [1])));
{"m":{"19723":1}}
```"#,
    argument(
        name = "struct_expr",
        description = "Struct expression to serialize, e.g. a struct column or a `named_struct` call. \
            The field names become the JSON keys. NULL returns NULL."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkToJson {
    signature: Signature,
}

impl Default for SparkToJson {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkToJson {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkToJson {
    fn name(&self) -> &str {
        "to_json"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match arg_types {
            [DataType::Struct(fields)] => {
                for field in fields {
                    check_spark_json_type(field.data_type(), field.name())?;
                }
                Ok(DataType::Utf8)
            }
            [other] => plan_err!("to_json expects a struct argument, got {other}"),
            _ => plan_err!(
                "to_json expects exactly one argument, got {}",
                arg_types.len()
            ),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let ScalarFunctionArgs {
            args,
            arg_fields,
            config_options,
            ..
        } = args;
        let [input] = take_function_args(self.name(), args)?;
        let [field] = take_function_args(self.name(), arg_fields)?;

        match input {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(struct_to_json(
                &field,
                &array,
                &config_options,
            )?)),
            ColumnarValue::Scalar(scalar) => {
                let array = scalar.to_array()?;
                let result = struct_to_json(&field, &array, &config_options)?;
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &result, 0,
                )?))
            }
        }
    }
}

/// Rejects types that have no faithful Spark JSON text. `path` names the
/// field in the error (`a.b[]` for the elements of list `b` in struct `a`).
fn check_spark_json_type(data_type: &DataType, path: &str) -> Result<()> {
    use DataType::*;
    match data_type {
        Null | Boolean | Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32
        | UInt64 | Float32 | Float64 | Decimal32(..) | Decimal64(..) | Decimal128(..)
        | Decimal256(..) | Utf8 | LargeUtf8 | Utf8View | Binary | LargeBinary
        | BinaryView | FixedSizeBinary(_) | Date32 | Date64 | Timestamp(..) => Ok(()),
        List(item)
        | LargeList(item)
        | FixedSizeList(item, _)
        | ListView(item)
        | LargeListView(item) => {
            check_spark_json_type(item.data_type(), &format!("{path}[]"))
        }
        Struct(fields) => fields.iter().try_for_each(|f| {
            check_spark_json_type(f.data_type(), &format!("{path}.{}", f.name()))
        }),
        Map(entries, _) => {
            let Struct(kv) = entries.data_type() else {
                return plan_err!("to_json: field {path} is a map without entries");
            };
            if !is_spark_map_key_type(kv[0].data_type()) {
                return plan_err!(
                    "to_json: field {path} has map key type {}, which is not supported",
                    kv[0].data_type()
                );
            }
            check_spark_json_type(kv[1].data_type(), &format!("{path}[]"))
        }
        // encodings rather than logical types: the values are what is written
        Dictionary(_, values) => check_spark_json_type(values, path),
        RunEndEncoded(_, values) => check_spark_json_type(values.data_type(), path),
        // Spark's interval text depends on the start and end fields (INTERVAL
        // '1' DAY vs INTERVAL '1 00:00:00' DAY TO SECOND), which Arrow does not
        // record; durations, times, Float16 and unions have no Spark type
        other => plan_err!(
            "to_json: field {path} has type {other}, which has no Spark JSON representation"
        ),
    }
}

/// Serializes every row of a struct array as a JSON object string; a NULL
/// struct row produces NULL. `field` describes `array` and supplies the JSON
/// keys of nested fields.
///
/// The arrow-json row encoder is built once per batch and each row is written
/// straight into a `StringBuilder`; the Spark-specific rules are supplied by
/// `SparkJsonEncoderFactory`.
fn struct_to_json(
    field: &FieldRef,
    array: &ArrayRef,
    config: &ConfigOptions,
) -> Result<ArrayRef> {
    // Spark writes TIMESTAMP values in the session time zone
    let session_tz: Tz = config
        .execution
        .time_zone
        .as_ref()
        .map(|tz| tz.as_str())
        .unwrap_or("+00:00")
        .parse()?;
    let options = EncoderOptions::default()
        // Spark default: ignoreNullFields = true -> NULL struct fields are omitted
        .with_explicit_nulls(false)
        .with_encoder_factory(Arc::new(SparkJsonEncoderFactory { session_tz }));

    let mut encoder = make_encoder(field, array.as_ref(), &options)?;
    let mut builder = StringBuilder::with_capacity(array.len(), array.len() * 32);
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    for row in 0..array.len() {
        if encoder.is_null(row) {
            builder.append_null();
            continue;
        }
        buf.clear();
        encoder.encode(row, &mut buf);
        let json = std::str::from_utf8(&buf)
            .map_err(|e| exec_datafusion_err!("to_json produced invalid UTF-8: {e}"))?;
        builder.append_value(json);
    }

    Ok(Arc::new(builder.finish()))
}

/// Overrides arrow-json's default encoders where Spark's `to_json` output
/// differs from arrow's. `make_encoder` consults the factory for every array at
/// every nesting level, so the overrides also apply inside lists, structs and
/// maps.
#[derive(Debug)]
struct SparkJsonEncoderFactory {
    /// `datafusion.execution.time_zone`; timestamps with a time zone are
    /// written in this zone
    session_tz: Tz,
}

impl EncoderFactory for SparkJsonEncoderFactory {
    fn make_default_encoder<'a>(
        &self,
        field: &'a FieldRef,
        array: &'a dyn Array,
        options: &'a EncoderOptions,
    ) -> Result<Option<NullableEncoder<'a>>, ArrowError> {
        match array.data_type() {
            DataType::Map(_, _) => {
                let array = array.as_map();
                let encoder = SparkMapEncoder::try_new(field, array, options)?;
                Ok(Some(NullableEncoder::new(
                    Box::new(encoder),
                    array.nulls().cloned(),
                )))
            }
            // Jackson writes control characters as \u001F, serde_json as \u001f
            DataType::Utf8 => Ok(Some(string_encoder(array.as_string::<i32>()))),
            DataType::LargeUtf8 => Ok(Some(string_encoder(array.as_string::<i64>()))),
            DataType::Utf8View => Ok(Some(string_encoder(array.as_string_view()))),
            // Spark serializes binary values as Base64 strings, arrow-json as hex
            DataType::Binary => Ok(Some(base64_encoder(array.as_binary::<i32>()))),
            DataType::LargeBinary => Ok(Some(base64_encoder(array.as_binary::<i64>()))),
            DataType::BinaryView => Ok(Some(base64_encoder(array.as_binary_view()))),
            DataType::FixedSizeBinary(_) => {
                Ok(Some(base64_encoder(array.as_fixed_size_binary())))
            }
            // Spark writes floats like Java's Double.toString (1.0E-5, 1.0) and
            // non-finite values as the strings "NaN", "Infinity" and "-Infinity";
            // arrow-json writes 0.00001 and JSON null
            DataType::Float32 => Ok(Some(spark_float_encoder(
                array.as_primitive::<Float32Type>(),
            ))),
            DataType::Float64 => Ok(Some(spark_float_encoder(
                array.as_primitive::<Float64Type>(),
            ))),
            // Spark writes decimals like Java's BigDecimal.toString (0E-10, 1.0E-7);
            // arrow-json writes them padded to the scale (0.0000000000). Spark data
            // maps to Decimal128 (precision <= 38); the other Arrow decimal widths
            // are formatted the same way rather than rejected.
            DataType::Decimal32(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal32Type>(),
            ))),
            DataType::Decimal64(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal64Type>(),
            ))),
            DataType::Decimal128(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal128Type>(),
            ))),
            DataType::Decimal256(_, _) => Ok(Some(spark_decimal_encoder(
                array.as_primitive::<Decimal256Type>(),
            ))),
            // Spark writes timestamps as yyyy-MM-dd'T'HH:mm:ss.SSS, instants in
            // the session time zone with their offset; arrow-json keeps the full
            // precision and uses the time zone recorded on the type
            DataType::Timestamp(unit, tz) => {
                let session_tz = tz.as_ref().map(|_| self.session_tz);
                Ok(Some(match unit {
                    TimeUnit::Second => timestamp_encoder(
                        array.as_primitive::<TimestampSecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Millisecond => timestamp_encoder(
                        array.as_primitive::<TimestampMillisecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Microsecond => timestamp_encoder(
                        array.as_primitive::<TimestampMicrosecondType>(),
                        session_tz,
                    )?,
                    TimeUnit::Nanosecond => timestamp_encoder(
                        array.as_primitive::<TimestampNanosecondType>(),
                        session_tz,
                    )?,
                }))
            }
            // Spark writes dates as yyyy-MM-dd; arrow-json writes a Date64 as a
            // datetime (a Date32 already comes out as a date)
            DataType::Date64 => {
                Ok(Some(date64_encoder(array.as_primitive::<Date64Type>())?))
            }
            // everything else uses arrow's default encoder
            _ => Ok(None),
        }
    }
}

/// Spark's default `timestampFormat`, `yyyy-MM-dd'T'HH:mm:ss[.SSS][XXX]`,
/// without the offset part (added separately so that UTC is written as `Z`).
const SPARK_TIMESTAMP_FORMAT: &str = "%Y-%m-%dT%H:%M:%S%.3f";

/// Builds an encoder that writes timestamps the way Spark does.
///
/// `session_tz` is `Some` for a timestamp with a time zone (a Spark
/// `TIMESTAMP`, an instant written in the session time zone with its offset)
/// and `None` for one without (`TIMESTAMP_NTZ`, written as is). The values
/// are formatted here, where an out-of-range value can be reported, so that
/// `encode` only has to copy them.
fn timestamp_encoder<'a, T: ArrowTimestampType>(
    array: &'a PrimitiveArray<T>,
    session_tz: Option<Tz>,
) -> Result<NullableEncoder<'a>, ArrowError> {
    let mut texts = StringBuilder::with_capacity(array.len(), array.len() * 30);
    for value in array.iter() {
        match value {
            Some(v) => texts.append_value(spark_timestamp_text::<T>(v, session_tz)?),
            None => texts.append_null(),
        }
    }
    Ok(NullableEncoder::new(
        Box::new(QuotedTextEncoder(texts.finish())),
        array.nulls().cloned(),
    ))
}

/// Formats one timestamp value as Spark's `to_json` does.
fn spark_timestamp_text<T: ArrowTimestampType>(
    value: i64,
    session_tz: Option<Tz>,
) -> Result<String, ArrowError> {
    let out_of_range = || {
        ArrowError::JsonError(format!(
            "to_json: timestamp value {value} ({:?}) is out of range",
            T::UNIT
        ))
    };
    match session_tz {
        Some(tz) => {
            let dt =
                as_datetime_with_timezone::<T>(value, tz).ok_or_else(out_of_range)?;
            let offset = dt.format("%:z").to_string();
            // Java's `XXX` pattern writes a zero offset as `Z`
            let offset = if offset == "+00:00" { "Z" } else { &offset };
            Ok(format!("{}{offset}", dt.format(SPARK_TIMESTAMP_FORMAT)))
        }
        None => Ok(as_datetime::<T>(value)
            .ok_or_else(out_of_range)?
            .format(SPARK_TIMESTAMP_FORMAT)
            .to_string()),
    }
}

/// Formats `Date64` values as `yyyy-MM-dd`.
fn date64_encoder(
    array: &PrimitiveArray<Date64Type>,
) -> Result<NullableEncoder<'_>, ArrowError> {
    let mut texts = StringBuilder::with_capacity(array.len(), array.len() * 10);
    for value in array.iter() {
        match value {
            Some(ms) => {
                let date = as_date::<Date64Type>(ms).ok_or_else(|| {
                    ArrowError::JsonError(format!(
                        "to_json: date value {ms} (ms) is out of range"
                    ))
                })?;
                texts.append_value(date.format("%Y-%m-%d").to_string());
            }
            None => texts.append_null(),
        }
    }
    Ok(NullableEncoder::new(
        Box::new(QuotedTextEncoder(texts.finish())),
        array.nulls().cloned(),
    ))
}

/// Writes pre-formatted text as a JSON string. The text must not need
/// escaping (timestamps and dates are digits and `-T:.+Z` only).
struct QuotedTextEncoder(StringArray);

impl Encoder for QuotedTextEncoder {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        out.push(b'"');
        out.extend_from_slice(self.0.value(idx).as_bytes());
        out.push(b'"');
    }
}

/// Floating point types that Spark serializes like Java's `Double.toString`.
trait SparkFloat: Copy + LowerExp {
    /// Java's text for a non-finite value (`NaN`, `Infinity`, `-Infinity`),
    /// or `None` if finite.
    fn non_finite_text(self) -> Option<&'static str>;
}

macro_rules! spark_float {
    ($($t:ty),*) => {$(
        impl SparkFloat for $t {
            fn non_finite_text(self) -> Option<&'static str> {
                if self.is_nan() {
                    Some("NaN")
                } else if self.is_infinite() {
                    Some(if self.is_sign_positive() {
                        "Infinity"
                    } else {
                        "-Infinity"
                    })
                } else {
                    None
                }
            }
        }
    )*};
}
spark_float!(f32, f64);

/// Formats a finite float the way Java's `Double.toString` / `Float.toString`
/// do, which is what Spark's `to_json` writes:
///
/// - `1e-3 <= |x| < 1e7`: plain decimal with at least one fraction digit
///   (`1.0`, `0.001`, `123.456`)
/// - otherwise: scientific notation `d.dddE<exp>` with at least one fraction
///   digit and no `+` in the exponent (`1.0E-5`, `1.23456789E8`, `1.0E10`)
///
/// The digits are the shortest that round-trip, taken from Rust's `{:e}`
/// formatting, which is also what Java produces (JDK 19+; older JDKs
/// occasionally emit one more digit).
fn java_float_text<F: LowerExp>(value: F) -> String {
    // e.g. "1.23456789e8", "1e-5", "-0e0"
    let exp_form = format!("{value:e}");
    let (sign, rest) = match exp_form.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", exp_form.as_str()),
    };
    // `{:e}` always has the form <digits>[.<digits>]e<exp>; if it ever did
    // not, the exponent form is still a valid JSON number
    let Some((mantissa, exp)) = rest.split_once('e') else {
        return exp_form;
    };
    let Ok(exp) = exp.parse::<i32>() else {
        return exp_form;
    };
    // all significant digits; `exp` is the decimal exponent of the first one
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();

    let mut text = String::with_capacity(digits.len() + 8);
    text.push_str(sign);
    if (-3..7).contains(&exp) {
        if exp < 0 {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
            text.push_str(&digits);
        } else {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                text.push_str(&digits);
                text.extend(std::iter::repeat_n('0', int_len - digits.len()));
                text.push_str(".0");
            } else {
                text.push_str(&digits[..int_len]);
                text.push('.');
                text.push_str(&digits[int_len..]);
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        text.push_str(first);
        text.push('.');
        text.push_str(if rest.is_empty() { "0" } else { rest });
        text.push('E');
        text.push_str(&exp.to_string());
    }
    text
}

/// Builds a float encoder that writes values the way Spark does.
fn spark_float_encoder<'a, T>(array: &'a PrimitiveArray<T>) -> NullableEncoder<'a>
where
    T: ArrowPrimitiveType,
    T::Native: SparkFloat,
{
    let encoder = SparkFloatEncoder {
        values: array.values(),
    };
    NullableEncoder::new(Box::new(encoder), array.nulls().cloned())
}

struct SparkFloatEncoder<'a, N: SparkFloat> {
    values: &'a [N],
}

impl<N: SparkFloat> Encoder for SparkFloatEncoder<'_, N> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let value = self.values[idx];
        match value.non_finite_text() {
            // Spark writes non-finite values as JSON strings
            Some(text) => {
                out.push(b'"');
                out.extend_from_slice(text.as_bytes());
                out.push(b'"');
            }
            None => out.extend_from_slice(java_float_text(value).as_bytes()),
        }
    }
}

/// Formats a decimal the way Java's `BigDecimal.toString` does, which is what
/// Spark's `to_json` writes. `unscaled` is the signed unscaled value as text.
///
/// With `adjusted = -scale + (digits - 1)`: plain notation when `scale >= 0`
/// and `adjusted >= -6` (`1.50`, `0.000001`), scientific notation otherwise
/// (`0E-10`, `1.0E-7`, `1.23E+4`; the exponent always carries a sign). Both
/// forms are valid JSON numbers, so they are written unquoted.
fn java_big_decimal_text(unscaled: &str, scale: i8) -> String {
    let (sign, coefficient) = match unscaled.strip_prefix('-') {
        Some(coefficient) => ("-", coefficient),
        None => ("", unscaled),
    };
    let adjusted = -(scale as i64) + (coefficient.len() as i64 - 1);

    let mut text = String::with_capacity(coefficient.len() + 8);
    text.push_str(sign);
    if scale >= 0 && adjusted >= -6 {
        let scale = scale as usize;
        if scale == 0 {
            text.push_str(coefficient);
        } else if coefficient.len() > scale {
            let (int, frac) = coefficient.split_at(coefficient.len() - scale);
            text.push_str(int);
            text.push('.');
            text.push_str(frac);
        } else {
            text.push_str("0.");
            text.extend(std::iter::repeat_n('0', scale - coefficient.len()));
            text.push_str(coefficient);
        }
    } else {
        // the coefficient of an integer has at least one digit
        let (first, rest) = coefficient.split_at(1);
        text.push_str(first);
        if !rest.is_empty() {
            text.push('.');
            text.push_str(rest);
        }
        text.push('E');
        if adjusted >= 0 {
            text.push('+');
        }
        text.push_str(&adjusted.to_string());
    }
    text
}

/// Builds a decimal encoder that writes values like Java's `BigDecimal.toString`.
/// The scale is the one declared on the Arrow type, as Spark's `BigDecimal`
/// carries the column's scale.
fn spark_decimal_encoder<'a, T>(array: &'a PrimitiveArray<T>) -> NullableEncoder<'a>
where
    T: DecimalType,
    T::Native: std::fmt::Display,
{
    let encoder = SparkDecimalEncoder {
        values: array.values(),
        scale: array.scale(),
    };
    NullableEncoder::new(Box::new(encoder), array.nulls().cloned())
}

struct SparkDecimalEncoder<'a, N> {
    values: &'a [N],
    scale: i8,
}

impl<N: std::fmt::Display + Copy> Encoder for SparkDecimalEncoder<'_, N> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let text = java_big_decimal_text(&self.values[idx].to_string(), self.scale);
        out.extend_from_slice(text.as_bytes());
    }
}

/// Builds a string encoder for any of arrow's string array types.
fn string_encoder<'a, S>(array: S) -> NullableEncoder<'a>
where
    S: ArrayAccessor<Item = &'a str> + Array + 'a,
{
    let nulls = array.nulls().cloned();
    NullableEncoder::new(Box::new(SparkStringEncoder(array)), nulls)
}

struct SparkStringEncoder<S>(S);

impl<'a, S: ArrayAccessor<Item = &'a str>> Encoder for SparkStringEncoder<S> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        write_json_string(self.0.value(idx), out);
    }
}

const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

/// Writes `s` as a JSON string the way Jackson (and so Spark) does: `"`, `\`
/// and the control characters are escaped, with the short forms for
/// `\b \t \n \f \r` and uppercase `\u00XX` for the rest.
fn write_json_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    let bytes = s.as_bytes();
    let mut start = 0;
    // only ASCII is matched: every byte of a multi-byte character is >= 0x80
    for (i, &byte) in bytes.iter().enumerate() {
        let short = match byte {
            b'"' => Some(b'"'),
            b'\\' => Some(b'\\'),
            0x08 => Some(b'b'),
            b'\t' => Some(b't'),
            b'\n' => Some(b'n'),
            0x0C => Some(b'f'),
            b'\r' => Some(b'r'),
            0x00..=0x1F => None,
            _ => continue,
        };
        out.extend_from_slice(&bytes[start..i]);
        match short {
            Some(c) => out.extend_from_slice(&[b'\\', c]),
            None => out.extend_from_slice(&[
                b'\\',
                b'u',
                b'0',
                b'0',
                HEX_UPPER[usize::from(byte >> 4)],
                HEX_UPPER[usize::from(byte & 0xF)],
            ]),
        }
        start = i + 1;
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}

/// Builds a Base64 encoder for any of arrow's binary array types.
fn base64_encoder<'a, B>(array: B) -> NullableEncoder<'a>
where
    B: ArrayAccessor<Item = &'a [u8]> + Array + 'a,
{
    let nulls = array.nulls().cloned();
    NullableEncoder::new(Box::new(Base64Encoder::new(array)), nulls)
}

/// Encodes binary values as Base64 strings (standard alphabet, padded), which
/// is what Spark's `to_json` produces; arrow-json's default is hexadecimal.
struct Base64Encoder<B> {
    array: B,
    scratch: String,
}

impl<B> Base64Encoder<B> {
    fn new(array: B) -> Self {
        Self {
            array,
            scratch: String::new(),
        }
    }
}

impl<'a, B> Encoder for Base64Encoder<B>
where
    B: ArrayAccessor<Item = &'a [u8]>,
{
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        self.scratch.clear();
        BASE64.encode_string(self.array.value(idx), &mut self.scratch);
        // Base64 output is [A-Za-z0-9+/=], so it needs no JSON escaping
        out.push(b'"');
        out.extend_from_slice(self.scratch.as_bytes());
        out.push(b'"');
    }
}

/// Encodes a map as a JSON object the way Spark does:
///
/// - NULL values are always written as `"key":null`. arrow-json's own map
///   encoder drops such entries when `explicit_nulls` is off, but Spark's
///   `ignoreNullFields` only applies to struct fields.
/// - keys are written as strings whatever their type (`{"1":"v"}`), see
///   `MapKeyEncoder`.
struct SparkMapEncoder<'a> {
    offsets: OffsetBuffer<i32>,
    keys: MapKeyEncoder,
    values: NullableEncoder<'a>,
}

impl<'a> SparkMapEncoder<'a> {
    fn try_new(
        field: &'a FieldRef,
        array: &'a MapArray,
        options: &'a EncoderOptions,
    ) -> Result<Self, ArrowError> {
        let keys = MapKeyEncoder::try_new(array.keys())?;
        // The value encoder is built with the same options, so nested structs
        // keep omitting NULL fields and nested maps use this encoder.
        let values = make_encoder(field, array.values(), options)?;

        Ok(Self {
            offsets: array.offsets().clone(),
            keys,
            values,
        })
    }
}

/// Encodes map keys as JSON strings holding the text that Spark gives them,
/// see `spark_map_key_texts`.
struct MapKeyEncoder(StringArray);

impl MapKeyEncoder {
    fn try_new(keys: &ArrayRef) -> Result<Self, ArrowError> {
        if keys.null_count() > 0 {
            return Err(ArrowError::InvalidArgumentError(
                "to_json: encountered a NULL map key".to_string(),
            ));
        }
        Ok(Self(spark_map_key_texts(keys)?))
    }
}

/// Stringifies map keys the way Spark does, with the `toString` of its
/// internal value: dates are days since the epoch, timestamps microseconds
/// since the epoch, floats and decimals follow Java's `Double.toString` and
/// `BigDecimal.toString`. Key types whose internal value has no meaningful
/// text (binary, lists, structs, maps, ...) are rejected.
fn spark_map_key_texts(keys: &ArrayRef) -> Result<StringArray, ArrowError> {
    use DataType::*;
    Ok(match keys.data_type() {
        // arrow's cast already produces Java's text for these
        Utf8 | LargeUtf8 | Utf8View | Boolean | Int8 | Int16 | Int32 | Int64 | UInt8
        | UInt16 | UInt32 | UInt64 => cast(keys, &Utf8)?.as_string::<i32>().clone(),
        Float32 => key_texts(keys.as_primitive::<Float32Type>(), java_float_key_text),
        Float64 => key_texts(keys.as_primitive::<Float64Type>(), java_float_key_text),
        // Spark's DateType holds the number of days since the epoch
        Date32 => key_texts(keys.as_primitive::<Date32Type>(), |days| days.to_string()),
        Date64 => key_texts(keys.as_primitive::<Date64Type>(), |ms| {
            ms.div_euclid(MILLISECONDS_IN_DAY).to_string()
        }),
        Timestamp(unit, _) => timestamp_key_texts(keys, *unit)?,
        Decimal32(_, scale) => decimal_key_texts::<Decimal32Type>(keys, *scale),
        Decimal64(_, scale) => decimal_key_texts::<Decimal64Type>(keys, *scale),
        Decimal128(_, scale) => decimal_key_texts::<Decimal128Type>(keys, *scale),
        Decimal256(_, scale) => decimal_key_texts::<Decimal256Type>(keys, *scale),
        other => {
            return Err(ArrowError::JsonError(format!(
                "to_json: unsupported map key type {other}"
            )));
        }
    })
}

/// The map key types `spark_map_key_texts` can stringify.
fn is_spark_map_key_type(data_type: &DataType) -> bool {
    use DataType::*;
    matches!(
        data_type,
        Utf8 | LargeUtf8
            | Utf8View
            | Boolean
            | Int8
            | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float32
            | Float64
            | Date32
            | Date64
            | Timestamp(..)
            | Decimal32(..)
            | Decimal64(..)
            | Decimal128(..)
            | Decimal256(..)
    )
}

/// Formats each key with `f` (NULL keys have been rejected already).
fn key_texts<T: ArrowPrimitiveType>(
    keys: &PrimitiveArray<T>,
    f: impl Fn(T::Native) -> String,
) -> StringArray {
    keys.iter().map(|key| key.map(&f)).collect()
}

/// Java's `Double.toString` for a map key: finite values as `java_float_text`,
/// non-finite ones as the bare words `NaN`, `Infinity` and `-Infinity`.
fn java_float_key_text<F: SparkFloat>(value: F) -> String {
    match value.non_finite_text() {
        Some(text) => text.to_string(),
        None => java_float_text(value),
    }
}

fn decimal_key_texts<T>(keys: &ArrayRef, scale: i8) -> StringArray
where
    T: DecimalType,
    T::Native: std::fmt::Display,
{
    key_texts(keys.as_primitive::<T>(), |unscaled| {
        java_big_decimal_text(&unscaled.to_string(), scale)
    })
}

/// Spark's TimestampType holds microseconds since the epoch, so timestamp
/// keys are written as such whatever their Arrow unit (and time zone).
fn timestamp_key_texts(
    keys: &ArrayRef,
    unit: TimeUnit,
) -> Result<StringArray, ArrowError> {
    match unit {
        TimeUnit::Second => {
            micros_key_texts(keys.as_primitive::<TimestampSecondType>(), |s| {
                s.checked_mul(MICROSECONDS)
            })
        }
        TimeUnit::Millisecond => {
            micros_key_texts(keys.as_primitive::<TimestampMillisecondType>(), |ms| {
                ms.checked_mul(MILLISECONDS)
            })
        }
        TimeUnit::Microsecond => {
            micros_key_texts(keys.as_primitive::<TimestampMicrosecondType>(), Some)
        }
        TimeUnit::Nanosecond => {
            micros_key_texts(keys.as_primitive::<TimestampNanosecondType>(), |ns| {
                Some(ns.div_euclid(NANOSECONDS / MICROSECONDS))
            })
        }
    }
}

/// Formats each key as the microseconds `to_micros` converts it to; a value
/// that does not fit is an error rather than a wrapped number.
fn micros_key_texts<T: ArrowTimestampType>(
    keys: &PrimitiveArray<T>,
    to_micros: impl Fn(i64) -> Option<i64>,
) -> Result<StringArray, ArrowError> {
    keys.iter()
        .map(|key| {
            key.map(|value| {
                to_micros(value).map(|us| us.to_string()).ok_or_else(|| {
                    ArrowError::JsonError(format!(
                        "to_json: timestamp map key {value} ({:?}) is out of range",
                        T::UNIT
                    ))
                })
            })
            .transpose()
        })
        .collect()
}

impl Encoder for MapKeyEncoder {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        write_json_string(self.0.value(idx), out);
    }
}

impl Encoder for SparkMapEncoder<'_> {
    fn encode(&mut self, idx: usize, out: &mut Vec<u8>) {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;

        out.push(b'{');
        for i in start..end {
            if i != start {
                out.push(b',');
            }
            self.keys.encode(i, out);
            out.push(b':');
            if self.values.is_null(i) {
                out.extend_from_slice(b"null");
            } else {
                self.values.encode(i, out);
            }
        }
        out.push(b'}');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        BinaryArray, BinaryBuilder, BinaryViewArray, BooleanArray, Date32Array,
        Date64Array, Decimal32Array, Decimal128Array, Decimal256Array, DictionaryArray,
        FixedSizeBinaryArray, Float32Array, Float64Array, Float64Builder, Int32Array,
        Int32Builder, Int64Array, LargeBinaryArray, LargeStringArray, ListArray,
        ListBuilder, MapBuilder, StringArray, StringViewArray, StructArray,
        TimestampMicrosecondArray, TimestampNanosecondArray, TimestampSecondArray,
        UInt8Array,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{
        Field, Fields, Int32Type, IntervalUnit, UnionFields, UnionMode, i256,
    };
    use datafusion_common::cast::as_string_array;
    use datafusion_common::config::ConfigOptions;

    /// Runs to_json over input (whose type is described by `field`) and
    /// returns the resulting string array.
    fn run(input: ColumnarValue, field: FieldRef) -> Result<ArrayRef> {
        run_with_config(input, field, ConfigOptions::default())
    }

    /// Like `run`, with the session configuration (e.g. the time zone).
    fn run_with_config(
        input: ColumnarValue,
        field: FieldRef,
        config: ConfigOptions,
    ) -> Result<ArrayRef> {
        let number_rows = match &input {
            ColumnarValue::Array(a) => a.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let args = ScalarFunctionArgs {
            args: vec![input],
            arg_fields: vec![field],
            number_rows,
            return_field: Field::new("to_json", DataType::Utf8, true).into(),
            config_options: Arc::new(config),
        };
        SparkToJson::new()
            .invoke_with_args(args)?
            .to_array(number_rows)
    }

    fn sample_struct() -> StructArray {
        let a = Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)])) as ArrayRef;
        let b = Arc::new(StringArray::from(vec![Some("x"), None, Some("z")])) as ArrayRef;
        let c =
            Arc::new(BooleanArray::from(vec![Some(true), Some(false), None])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Boolean, true),
        ]);
        let nulls = NullBuffer::from(vec![true, true, false]);
        StructArray::new(fields, vec![a, b, c], Some(nulls))
    }

    fn field_of(array: &StructArray) -> FieldRef {
        Field::new("s", array.data_type().clone(), true).into()
    }

    #[test]
    fn test_return_type() {
        let func = SparkToJson::new();
        let fields = Fields::from(vec![Field::new("a", DataType::Int32, true)]);
        assert_eq!(
            func.return_type(&[DataType::Struct(fields)]).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn test_return_type_accepts_types_with_spark_json_text() {
        let item = |t: DataType| Arc::new(Field::new("item", t, true));
        let map_of = |k: DataType, v: DataType| {
            let entries = Fields::from(vec![
                Field::new("keys", k, false),
                Field::new("values", v, true),
            ]);
            DataType::Map(
                Arc::new(Field::new("entries", DataType::Struct(entries), false)),
                false,
            )
        };
        let types = vec![
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Decimal128(10, 2),
            DataType::Decimal256(50, 2),
            DataType::Utf8View,
            DataType::FixedSizeBinary(3),
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::List(item(DataType::Date32)),
            DataType::FixedSizeList(item(DataType::Float64), 2),
            DataType::Struct(Fields::from(vec![Field::new("x", DataType::Utf8, true)])),
            map_of(DataType::Date32, DataType::List(item(DataType::Int32))),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                item(DataType::Float64),
            ),
        ];
        let fields: Fields = types
            .into_iter()
            .enumerate()
            .map(|(i, t)| Field::new(format!("f{i}"), t, true))
            .collect();
        assert_eq!(
            SparkToJson::new()
                .return_type(&[DataType::Struct(fields)])
                .unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn test_return_type_rejects_types_without_spark_json_text() {
        let return_type = |name: &str, t: DataType| {
            let fields = Fields::from(vec![Field::new(name, t, true)]);
            SparkToJson::new()
                .return_type(&[DataType::Struct(fields)])
                .unwrap_err()
                .to_string()
        };
        let item = |t: DataType| Arc::new(Field::new("item", t, true));
        let union = DataType::Union(
            UnionFields::from_fields(vec![Field::new("a", DataType::Int32, true)]),
            UnionMode::Sparse,
        );

        for t in [
            DataType::Interval(IntervalUnit::YearMonth),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Duration(TimeUnit::Second),
            DataType::Time32(TimeUnit::Second),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Float16,
            union,
        ] {
            let err = return_type("x", t.clone());
            assert!(
                err.contains(&format!(
                    "field x has type {t}, which has no Spark JSON representation"
                )),
                "unexpected error for {t}: {err}"
            );
        }

        // nested types report the path to the offending field
        let interval = DataType::Interval(IntervalUnit::DayTime);
        let err = return_type("l", DataType::List(item(interval.clone())));
        assert!(err.contains("field l[] has type"), "{err}");

        let inner = Fields::from(vec![Field::new("i", interval.clone(), true)]);
        let err = return_type("s", DataType::Struct(inner));
        assert!(err.contains("field s.i has type"), "{err}");

        let entries = Fields::from(vec![
            Field::new("keys", DataType::Utf8, false),
            Field::new("values", interval.clone(), true),
        ]);
        let entries = Arc::new(Field::new("entries", DataType::Struct(entries), false));
        let err = return_type("m", DataType::Map(entries, false));
        assert!(err.contains("field m[] has type"), "{err}");

        let err = return_type(
            "d",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(interval)),
        );
        assert!(err.contains("field d has type Interval(DayTime)"), "{err}");

        // map keys are checked too (lists have no Spark key text)
        let entries = Fields::from(vec![
            Field::new("keys", DataType::List(item(DataType::Int32)), false),
            Field::new("values", DataType::Int32, true),
        ]);
        let entries = Arc::new(Field::new("entries", DataType::Struct(entries), false));
        let err = return_type("m", DataType::Map(entries, false));
        assert!(
            err.contains("field m has map key type List(")
                && err.contains("not supported"),
            "{err}"
        );
    }

    #[test]
    fn test_dates_are_written_as_yyyy_mm_dd() {
        // 2024-01-01 is day 19723; the Date64 is one millisecond into that day
        let d32 = Arc::new(Date32Array::from(vec![Some(19723), None])) as ArrayRef;
        let d64 = Arc::new(Date64Array::from(vec![
            Some(19723 * MILLISECONDS_IN_DAY + 1),
            None,
        ])) as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("d32", DataType::Date32, true),
            Field::new("d64", DataType::Date64, true),
        ]);
        let input = StructArray::new(fields, vec![d32, d64], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(out.value(0), r#"{"d32":"2024-01-01","d64":"2024-01-01"}"#);
        assert_eq!(out.value(1), "{}");
    }

    #[test]
    fn test_dictionary_values_use_spark_encoders() {
        // Dictionary<Int32, Float64> [1e10, 1e10]: the float rule applies to the values
        let values = Arc::new(Float64Array::from(vec![1e10])) as ArrayRef;
        let dict =
            DictionaryArray::<Int32Type>::new(Int32Array::from(vec![0, 0]), values);
        let dict = Arc::new(dict) as ArrayRef;
        let fields = Fields::from(vec![Field::new("x", dict.data_type().clone(), true)]);
        let input = StructArray::new(fields, vec![dict], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(out.value(0), r#"{"x":1.0E10}"#);
        assert_eq!(out.value(1), r#"{"x":1.0E10}"#);
    }

    #[test]
    fn test_return_type_rejects_non_struct() {
        let err = SparkToJson::new()
            .return_type(&[DataType::Utf8])
            .unwrap_err();
        assert!(err.to_string().contains("expects a struct argument"));

        let err = SparkToJson::new().return_type(&[]).unwrap_err();
        assert!(err.to_string().contains("exactly one argument"));
    }

    #[test]
    fn test_array_input() {
        let input = sample_struct();
        let field = field_of(&input);
        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out.value(0), r#"{"a":1,"b":"x","c":true}"#);
        // NULL field is omitted (Spark ignoreNullFields = true)
        assert_eq!(out.value(1), r#"{"a":2,"c":false}"#);
        // NULL struct row -> NULL
        assert!(out.is_null(2));
    }

    #[test]
    fn test_scalar_input() {
        let input = sample_struct();
        let field = field_of(&input);
        let scalar = ScalarValue::Struct(Arc::new(input.slice(0, 1)));
        let out = run(ColumnarValue::Scalar(scalar), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out.value(0), r#"{"a":1,"b":"x","c":true}"#);
    }

    #[test]
    fn test_scalar_null_struct() {
        let input = sample_struct();
        let field = field_of(&input);
        let scalar = ScalarValue::Struct(Arc::new(input.slice(2, 1)));
        let out = run(ColumnarValue::Scalar(scalar), field).unwrap();

        assert_eq!(out.len(), 1);
        assert!(out.is_null(0));
    }

    #[test]
    fn test_nested_struct_and_list() {
        // {inner: {x: Int64}, tags: List<Utf8>}
        let x = Arc::new(Int64Array::from(vec![Some(10), None])) as ArrayRef;
        let inner_fields = Fields::from(vec![Field::new("x", DataType::Int64, true)]);
        let inner = Arc::new(StructArray::new(inner_fields, vec![x], None)) as ArrayRef;

        let mut tags = ListBuilder::new(StringBuilder::new());
        tags.values().append_value("p");
        tags.values().append_value("q");
        tags.append(true);
        tags.append(false);
        let tags = Arc::new(tags.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("inner", inner.data_type().clone(), true),
            Field::new("tags", tags.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![inner, tags], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"inner":{"x":10},"tags":["p","q"]}"#);
        // nested NULL field and NULL list are both omitted
        assert_eq!(out.value(1), r#"{"inner":{}}"#);
    }

    #[test]
    fn test_sliced_nested_struct_and_list() {
        // 4 rows of {inner: {x: Int64}, tags: List<Utf8>} with uneven list
        // lengths (non-trivial list offsets) and a NULL parent in row 2:
        //   0: {x:1}, []        1: {x:2}, [a,b]
        //   2: NULL             3: {x:4}, [d,e,f]
        let x = Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let inner_fields = Fields::from(vec![Field::new("x", DataType::Int64, true)]);
        let inner = Arc::new(StructArray::new(inner_fields, vec![x], None)) as ArrayRef;

        let mut tags = ListBuilder::new(StringBuilder::new());
        tags.append(true); // []
        for t in ["a", "b"] {
            tags.values().append_value(t);
        }
        tags.append(true);
        tags.values().append_value("c");
        tags.append(true);
        for t in ["d", "e", "f"] {
            tags.values().append_value(t);
        }
        tags.append(true);
        let tags = Arc::new(tags.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("inner", inner.data_type().clone(), true),
            Field::new("tags", tags.data_type().clone(), true),
        ]);
        let nulls = NullBuffer::from(vec![true, true, false, true]);
        let full = StructArray::new(fields, vec![inner, tags], Some(nulls));
        let field = field_of(&full);

        // encode rows 1..4 through a non-zero-offset slice
        let sliced = full.slice(1, 3);
        let out = run(ColumnarValue::Array(Arc::new(sliced)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.len(), 3);
        assert_eq!(out.null_count(), 1);
        assert_eq!(out.value(0), r#"{"inner":{"x":2},"tags":["a","b"]}"#);
        assert!(
            out.is_null(1),
            "NULL parent row must stay NULL after slicing"
        );
        assert_eq!(out.value(2), r#"{"inner":{"x":4},"tags":["d","e","f"]}"#);
    }

    #[test]
    fn test_string_escaping() {
        let a = Arc::new(StringArray::from(vec![Some("he said \"hi\"\n")])) as ArrayRef;
        let fields = Fields::from(vec![Field::new("a", DataType::Utf8, true)]);
        let input = StructArray::new(fields, vec![a], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"a":"he said \"hi\"\n"}"#);
    }

    #[test]
    fn test_map_null_values_kept_but_null_struct_fields_omitted() {
        // m: Map<Utf8, Int32> with rows {"k":null}, {"a":1,"b":null}, NULL
        let mut mb = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        mb.keys().append_value("k");
        mb.values().append_null();
        mb.append(true).unwrap();
        mb.keys().append_value("a");
        mb.values().append_value(1);
        mb.keys().append_value("b");
        mb.values().append_null();
        mb.append(true).unwrap();
        mb.append(false).unwrap();
        let m = Arc::new(mb.finish()) as ArrayRef;

        // s: Int32 with a NULL in row 1
        let s = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("m", m.data_type().clone(), true),
            Field::new("s", DataType::Int32, true),
        ]);
        let input = StructArray::new(fields, vec![m, s], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        // a NULL map *value* is written explicitly (Spark keeps map entries)
        assert_eq!(out.value(0), r#"{"m":{"k":null},"s":1}"#);
        // ...while a NULL struct *field* (`s`) is still omitted
        assert_eq!(out.value(1), r#"{"m":{"a":1,"b":null}}"#);
        // a NULL map is a NULL struct field, so it is omitted as well
        assert_eq!(out.value(2), r#"{"s":3}"#);
    }

    #[test]
    fn test_binary_is_base64_encoded() {
        // b: Binary ["abc", "", NULL] -> "YWJj", "", omitted
        let b = Arc::new(BinaryArray::from(vec![
            Some(b"abc".as_slice()),
            Some(b"".as_slice()),
            None,
        ])) as ArrayRef;
        let abc = vec![Some(b"abc".as_slice()); 3];
        let lb = Arc::new(LargeBinaryArray::from(abc.clone())) as ArrayRef;
        let bv = Arc::new(BinaryViewArray::from(abc)) as ArrayRef;
        let fsb = Arc::new(
            FixedSizeBinaryArray::try_from_iter(vec![b"ab".as_slice(); 3].into_iter())
                .unwrap(),
        ) as ArrayRef;

        // l: List<Binary> [["abc"]] per row; n: Struct<b: Binary> {"abc"} per row
        let mut l = ListBuilder::new(BinaryBuilder::new());
        for _ in 0..3 {
            l.values().append_value(b"abc");
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("b", DataType::LargeBinary, true)]),
            vec![Arc::clone(&lb)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("b", DataType::Binary, true),
            Field::new("lb", DataType::LargeBinary, true),
            Field::new("bv", DataType::BinaryView, true),
            Field::new("fsb", DataType::FixedSizeBinary(2), true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![b, lb, bv, fsb, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"b":"YWJj","lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
        // empty binary is an empty string, not NULL
        assert_eq!(
            out.value(1),
            r#"{"b":"","lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
        // NULL binary is a NULL struct field -> omitted
        assert_eq!(
            out.value(2),
            r#"{"lb":"YWJj","bv":"YWJj","fsb":"YWI=","l":["YWJj"],"n":{"b":"YWJj"}}"#
        );
    }

    #[test]
    fn test_non_finite_floats_are_spark_strings() {
        // d: Float64 [NaN, +inf, -inf, 1.5, NULL]; f: Float32 the same
        let d = Arc::new(Float64Array::from(vec![
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(1.5),
            None,
        ])) as ArrayRef;
        let f = Arc::new(Float32Array::from(vec![
            Some(f32::NAN),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(1.5),
            None,
        ])) as ArrayRef;

        // l: List<Float64> [[NaN, 1.5]] per row; n: Struct<d: Float64>
        let mut l = ListBuilder::new(Float64Builder::new());
        for _ in 0..5 {
            l.values().append_value(f64::NAN);
            l.values().append_value(1.5);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("d", DataType::Float64, true)]),
            vec![Arc::clone(&d)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, f, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":"NaN","f":"NaN","l":["NaN",1.5],"n":{"d":"NaN"}}"#
        );
        assert_eq!(
            out.value(1),
            r#"{"d":"Infinity","f":"Infinity","l":["NaN",1.5],"n":{"d":"Infinity"}}"#
        );
        assert_eq!(
            out.value(2),
            r#"{"d":"-Infinity","f":"-Infinity","l":["NaN",1.5],"n":{"d":"-Infinity"}}"#
        );
        // finite values keep arrow's formatting
        assert_eq!(
            out.value(3),
            r#"{"d":1.5,"f":1.5,"l":["NaN",1.5],"n":{"d":1.5}}"#
        );
        // NULL is still a NULL struct field -> omitted
        assert_eq!(out.value(4), r#"{"l":["NaN",1.5],"n":{}}"#);
    }

    #[test]
    fn test_write_json_string() {
        let json = |s: &str| {
            let mut out = Vec::new();
            write_json_string(s, &mut out);
            String::from_utf8(out).unwrap()
        };
        assert_eq!(json(""), "\"\"");
        assert_eq!(json("plain text"), r#""plain text""#);
        assert_eq!(json(r#"q"b\"#), r#""q\"b\\""#);
        assert_eq!(json("\u{8}\t\n\u{c}\r"), r#""\b\t\n\f\r""#);
        // other control characters use uppercase hex, as Jackson does
        assert_eq!(json("a\u{1f}b"), r#""a\u001Fb""#);
        assert_eq!(json("\u{0}\u{1}\u{1b}"), r#""\u0000\u0001\u001B""#);
        // `/`, DEL and non-ASCII are written as is
        assert_eq!(json("a/b\u{7f}é🦀"), "\"a/b\u{7f}é🦀\"");
    }

    #[test]
    fn test_strings_are_escaped_like_jackson() {
        let text = "a\u{1f}\"b";
        let s = Arc::new(StringArray::from(vec![Some(text), None])) as ArrayRef;
        let l = Arc::new(LargeStringArray::from(vec![Some(text), None])) as ArrayRef;
        let v = Arc::new(StringViewArray::from(vec![Some(text), None])) as ArrayRef;

        let mut m = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
        m.keys().append_value(text);
        m.values().append_value(text);
        m.append(true).unwrap();
        m.append(true).unwrap();
        let m = Arc::new(m.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("l", DataType::LargeUtf8, true),
            Field::new("v", DataType::Utf8View, true),
            Field::new("m", m.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![s, l, v, m], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"s":"a\u001F\"b","l":"a\u001F\"b","v":"a\u001F\"b","m":{"a\u001F\"b":"a\u001F\"b"}}"#
        );
        assert_eq!(out.value(1), r#"{"m":{}}"#);
    }

    #[test]
    fn test_map_keys_are_stringified() {
        // i: Map<Int32, Utf8> {1:"a", 2:"b"}; f: Map<Float64, Utf8> {1.5:"c"}
        let mut i = MapBuilder::new(None, Int32Builder::new(), StringBuilder::new());
        i.keys().append_value(1);
        i.values().append_value("a");
        i.keys().append_value(2);
        i.values().append_value("b");
        i.append(true).unwrap();
        let i = Arc::new(i.finish()) as ArrayRef;

        let mut f = MapBuilder::new(None, Float64Builder::new(), StringBuilder::new());
        f.keys().append_value(1.5);
        f.values().append_value("c");
        f.append(true).unwrap();
        let f = Arc::new(f.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("i", i.data_type().clone(), true),
            Field::new("f", f.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![i, f], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(out.value(0), r#"{"i":{"1":"a","2":"b"},"f":{"1.5":"c"}}"#);
    }

    /// Serializes the one-row struct `{"m": map}` whose map pairs `keys` with
    /// the values 1, 2, ...
    fn map_with_keys_json(keys: ArrayRef) -> Result<String> {
        let n = keys.len();
        let values = Arc::new(Int32Array::from_iter_values(1..=n as i32)) as ArrayRef;
        let entry_fields = Fields::from(vec![
            Field::new("keys", keys.data_type().clone(), false),
            Field::new("values", DataType::Int32, true),
        ]);
        let entries = StructArray::new(entry_fields.clone(), vec![keys, values], None);
        let map = MapArray::new(
            Arc::new(Field::new("entries", DataType::Struct(entry_fields), false)),
            OffsetBuffer::new(vec![0, n as i32].into()),
            entries,
            None,
            false,
        );
        let m = Arc::new(map) as ArrayRef;

        let fields = Fields::from(vec![Field::new("m", m.data_type().clone(), true)]);
        let input = StructArray::new(fields, vec![m], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field)?;
        Ok(as_string_array(&out)?.value(0).to_string())
    }

    #[test]
    fn test_map_keys_use_spark_internal_values() {
        let json = |keys: ArrayRef| map_with_keys_json(keys).unwrap();

        // integers of any width keep their text
        assert_eq!(
            json(Arc::new(Int64Array::from(vec![-7, 42]))),
            r#"{"m":{"-7":1,"42":2}}"#
        );
        assert_eq!(
            json(Arc::new(UInt8Array::from(vec![255]))),
            r#"{"m":{"255":1}}"#
        );

        // floats like Java's Double.toString, non-finite as bare words
        assert_eq!(
            json(Arc::new(Float64Array::from(vec![
                1e10,
                1.5,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ]))),
            r#"{"m":{"1.0E10":1,"1.5":2,"NaN":3,"Infinity":4,"-Infinity":5}}"#
        );
        assert_eq!(
            json(Arc::new(Float32Array::from(vec![1e10f32, 0.001]))),
            r#"{"m":{"1.0E10":1,"0.001":2}}"#
        );

        // dates are days since the epoch (2024-01-01 is day 19723)
        assert_eq!(
            json(Arc::new(Date32Array::from(vec![19723, -1]))),
            r#"{"m":{"19723":1,"-1":2}}"#
        );
        assert_eq!(
            json(Arc::new(Date64Array::from(vec![
                19723 * MILLISECONDS_IN_DAY + 1,
                -1,
            ]))),
            r#"{"m":{"19723":1,"-1":2}}"#
        );

        // timestamps are microseconds since the epoch, whatever the Arrow
        // unit; the time zone does not change the instant
        assert_eq!(
            json(Arc::new(
                TimestampMicrosecondArray::from(vec![TS_MICROS]).with_timezone("+02:00")
            )),
            r#"{"m":{"1704070923123456":1}}"#
        );
        assert_eq!(
            json(Arc::new(TimestampSecondArray::from(vec![
                TS_MICROS / 1_000_000
            ]))),
            r#"{"m":{"1704070923000000":1}}"#
        );
        assert_eq!(
            json(Arc::new(TimestampNanosecondArray::from(vec![
                TS_MICROS * 1_000 + 789,
            ]))),
            r#"{"m":{"1704070923123456":1}}"#
        );

        // decimals like Java's BigDecimal.toString
        assert_eq!(
            json(Arc::new(
                Decimal128Array::from(vec![10, 150_000_000])
                    .with_precision_and_scale(10, 8)
                    .unwrap()
            )),
            r#"{"m":{"1.0E-7":1,"1.50000000":2}}"#
        );
        assert_eq!(
            json(Arc::new(
                Decimal32Array::from(vec![150])
                    .with_precision_and_scale(5, 2)
                    .unwrap()
            )),
            r#"{"m":{"1.50":1}}"#
        );
    }

    #[test]
    fn test_timestamp_map_key_out_of_range_errors() {
        let keys = Arc::new(TimestampSecondArray::from(vec![i64::MAX])) as ArrayRef;
        let err = map_with_keys_json(keys).unwrap_err();
        assert!(
            err.to_string().contains("timestamp map key")
                && err.to_string().contains("is out of range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_map_with_unsupported_key_type_errors() {
        // Spark would print an object dump for these, so there is no faithful text
        let list_keys = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1)]),
        ])) as ArrayRef;
        let binary_keys = Arc::new(BinaryArray::from(vec![b"k".as_ref()])) as ArrayRef;
        let struct_keys = Arc::new(StructArray::from(vec![(
            Arc::new(Field::new("a", DataType::Int32, true)),
            Arc::new(Int32Array::from(vec![1])) as ArrayRef,
        )])) as ArrayRef;

        for keys in [list_keys, binary_keys, struct_keys] {
            let err = map_with_keys_json(keys).unwrap_err();
            assert!(
                err.to_string()
                    .contains("to_json: unsupported map key type"),
                "unexpected error: {err}"
            );
        }
    }

    /// 2024-01-01 01:02:03.123456 UTC, in microseconds since the epoch
    const TS_MICROS: i64 = 1_704_070_923_123_456;

    #[test]
    fn test_timestamps_use_spark_format_and_session_time_zone() {
        // ltz: Timestamp(us, "UTC") -> Spark TIMESTAMP (an instant)
        // ntz: Timestamp(us, None)  -> Spark TIMESTAMP_NTZ (a wall-clock time)
        let ltz = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(TS_MICROS), None])
                .with_timezone("UTC"),
        ) as ArrayRef;
        let ntz = Arc::new(TimestampMicrosecondArray::from(vec![Some(TS_MICROS), None]))
            as ArrayRef;
        let fields = Fields::from(vec![
            Field::new("ltz", ltz.data_type().clone(), true),
            Field::new("ntz", ntz.data_type().clone(), true),
        ]);
        let input = Arc::new(StructArray::new(fields, vec![ltz, ntz], None));
        let field = field_of(&input);

        // default session time zone (UTC): milliseconds, offset written as `Z`
        let out = run(
            ColumnarValue::Array(Arc::clone(&input) as ArrayRef),
            Arc::clone(&field),
        )
        .unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"ltz":"2024-01-01T01:02:03.123Z","ntz":"2024-01-01T01:02:03.123"}"#
        );
        // NULL timestamps are NULL struct fields -> omitted
        assert_eq!(out.value(1), "{}");

        // the instant is written in the session time zone with its offset;
        // the wall-clock value is unchanged
        let mut config = ConfigOptions::default();
        config
            .set("datafusion.execution.time_zone", "+02:00")
            .unwrap();
        let out = run_with_config(ColumnarValue::Array(input), field, config).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"ltz":"2024-01-01T03:02:03.123+02:00","ntz":"2024-01-01T01:02:03.123"}"#
        );
    }

    #[test]
    fn test_timestamp_type_time_zone_is_ignored_and_precision_is_millis() {
        // Spark has no per-column time zone: the zone recorded on the Arrow
        // type does not change the output, only the session time zone does
        let plus5 = Arc::new(
            TimestampMicrosecondArray::from(vec![Some(TS_MICROS)])
                .with_timezone("+05:00"),
        ) as ArrayRef;
        // whole seconds still get three fractional digits; a nested list too
        let secs = Arc::new(
            TimestampSecondArray::from(vec![Some(TS_MICROS / 1_000_000)])
                .with_timezone("UTC"),
        ) as ArrayRef;
        let mut l =
            ListBuilder::new(TimestampSecondArray::builder(1).with_timezone("UTC"));
        l.values().append_value(TS_MICROS / 1_000_000);
        l.append(true);
        let l = Arc::new(l.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("plus5", plus5.data_type().clone(), true),
            Field::new("secs", secs.data_type().clone(), true),
            Field::new("l", l.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![plus5, secs, l], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();
        assert_eq!(
            out.value(0),
            r#"{"plus5":"2024-01-01T01:02:03.123Z","secs":"2024-01-01T01:02:03.000Z","l":["2024-01-01T01:02:03.000Z"]}"#
        );
    }

    #[test]
    fn test_java_float_text() {
        // Java Double.toString: plain decimal for 1e-3 <= |x| < 1e7, else scientific
        let cases: [(f64, &str); 14] = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (123.456, "123.456"),
            (0.001, "0.001"),
            (0.0001, "1.0E-4"),
            (1e-5, "1.0E-5"),
            (9999999.0, "9999999.0"),
            (1e7, "1.0E7"),
            (123456789.0, "1.23456789E8"),
            (1e10, "1.0E10"),
            (-2.5e-8, "-2.5E-8"),
            (f64::MAX, "1.7976931348623157E308"),
        ];
        for (value, expected) in cases {
            assert_eq!(java_float_text(value), expected, "value {value:?}");
        }
        // Float.toString uses the shortest digits of the f32 value
        let cases: [(f32, &str); 4] = [
            (0.1, "0.1"),
            (1e10, "1.0E10"),
            (3.0, "3.0"),
            (f32::MAX, "3.4028235E38"),
        ];
        for (value, expected) in cases {
            assert_eq!(java_float_text(value), expected, "value {value:?}");
        }
    }

    #[test]
    fn test_finite_floats_use_java_to_string() {
        // d: Float64, f: Float32, l: List<Float64>, n: Struct<d: Float64>
        let d =
            Arc::new(Float64Array::from(vec![1e-5, 123456789.0, 1e10, 1.5])) as ArrayRef;
        let f = Arc::new(Float32Array::from(vec![0.1, 1e10, 3.0, 1.5])) as ArrayRef;
        let mut l = ListBuilder::new(Float64Builder::new());
        for _ in 0..4 {
            l.values().append_value(1e-5);
            l.values().append_value(1.0);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;
        let n = Arc::new(StructArray::new(
            Fields::from(vec![Field::new("d", DataType::Float64, true)]),
            vec![Arc::clone(&d)],
            None,
        )) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("f", DataType::Float32, true),
            Field::new("l", l.data_type().clone(), true),
            Field::new("n", n.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, f, l, n], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":1.0E-5,"f":0.1,"l":[1.0E-5,1.0],"n":{"d":1.0E-5}}"#
        );
        assert_eq!(
            out.value(1),
            r#"{"d":1.23456789E8,"f":1.0E10,"l":[1.0E-5,1.0],"n":{"d":1.23456789E8}}"#
        );
        assert_eq!(
            out.value(2),
            r#"{"d":1.0E10,"f":3.0,"l":[1.0E-5,1.0],"n":{"d":1.0E10}}"#
        );
        assert_eq!(
            out.value(3),
            r#"{"d":1.5,"f":1.5,"l":[1.0E-5,1.0],"n":{"d":1.5}}"#
        );
    }

    #[test]
    fn test_java_big_decimal_text() {
        // Java BigDecimal.toString: plain when scale >= 0 and adjusted >= -6,
        // scientific (signed exponent) otherwise
        let cases: [(&str, i8, &str); 13] = [
            ("0", 10, "0E-10"),  // 0 as decimal(38,10)
            ("10", 8, "1.0E-7"), // 0.0000001 as decimal(10,8)
            ("150", 2, "1.50"),
            ("123", 0, "123"),
            ("1", 6, "0.000001"), // adjusted = -6: still plain
            ("1", 7, "1E-7"),     // adjusted = -7: scientific, single digit
            ("-5", 2, "-0.05"),
            ("0", 2, "0.00"),
            ("0", 0, "0"),
            ("-123", 8, "-0.00000123"), // adjusted = -6: still plain
            ("-123", 9, "-1.23E-7"),    // adjusted = -7: scientific
            ("123", -2, "1.23E+4"),     // negative scale: exponent gets a '+'
            (
                "12345678901234567890123456789012345678",
                10,
                "1234567890123456789012345678.9012345678",
            ),
        ];
        for (unscaled, scale, expected) in cases {
            assert_eq!(
                java_big_decimal_text(unscaled, scale),
                expected,
                "unscaled {unscaled} scale {scale}"
            );
        }
    }

    #[test]
    fn test_decimals_use_java_big_decimal_to_string() {
        // d: Decimal128(38,10) [0, 1.5], e: Decimal128(10,8) [0.0000001, NULL],
        // s: Decimal32(5,2) [1.50], w: Decimal256(50,3) [-0.005],
        // l: List<Decimal128(10,8)> [[0.0000001]]
        let d = Arc::new(
            Decimal128Array::from(vec![Some(0), Some(15_000_000_000)])
                .with_precision_and_scale(38, 10)
                .unwrap(),
        ) as ArrayRef;
        let e = Arc::new(
            Decimal128Array::from(vec![Some(10), None])
                .with_precision_and_scale(10, 8)
                .unwrap(),
        ) as ArrayRef;
        let s = Arc::new(
            Decimal32Array::from(vec![Some(150), Some(150)])
                .with_precision_and_scale(5, 2)
                .unwrap(),
        ) as ArrayRef;
        let w = Arc::new(
            Decimal256Array::from(vec![Some(i256::from(-5)), Some(i256::from(-5))])
                .with_precision_and_scale(50, 3)
                .unwrap(),
        ) as ArrayRef;
        let mut l = ListBuilder::new(
            Decimal128Array::builder(1)
                .with_precision_and_scale(10, 8)
                .unwrap(),
        );
        for _ in 0..2 {
            l.values().append_value(10);
            l.append(true);
        }
        let l = Arc::new(l.finish()) as ArrayRef;

        let fields = Fields::from(vec![
            Field::new("d", d.data_type().clone(), true),
            Field::new("e", e.data_type().clone(), true),
            Field::new("s", s.data_type().clone(), true),
            Field::new("w", w.data_type().clone(), true),
            Field::new("l", l.data_type().clone(), true),
        ]);
        let input = StructArray::new(fields, vec![d, e, s, w, l], None);
        let field = field_of(&input);

        let out = run(ColumnarValue::Array(Arc::new(input)), field).unwrap();
        let out = as_string_array(&out).unwrap();

        assert_eq!(
            out.value(0),
            r#"{"d":0E-10,"e":1.0E-7,"s":1.50,"w":-0.005,"l":[1.0E-7]}"#
        );
        // 1.5 keeps the declared scale; a NULL decimal is omitted
        assert_eq!(
            out.value(1),
            r#"{"d":1.5000000000,"s":1.50,"w":-0.005,"l":[1.0E-7]}"#
        );
    }
}

/// DataFrame-level tests: to_json used through DataFrame::with_column
/// and through SQL, the way a user of the crate calls it.
#[cfg(test)]
mod dataframe_tests {
    use super::*;

    use datafusion::dataframe;
    use datafusion::prelude::*;
    use datafusion_common::cast::as_string_array;
    use datafusion_common::test_util::batches_to_sort_string;
    use datafusion_common::{assert_batches_sorted_eq, assert_contains};
    use datafusion_functions_nested::expr_fn::make_array;

    use crate::function::json::expr_fn::to_json;

    /// id  foo  bar
    /// 10  1    NULL
    /// 20  2    bak
    /// 30  3    baz
    fn sample_frame() -> Result<DataFrame> {
        dataframe!(
            "id" => [10_i32, 20, 30],
            "foo" => [1_i32, 2, 3],
            "bar" => [None, Some("bak"), Some("baz")],
        )
    }

    /// The same rows as `sample_frame`, as a SQL table expression named `t`.
    const SAMPLE_TABLE_SQL: &str =
        "(VALUES (20, 2, 'bak'), (30, 3, 'baz'), (10, 1, NULL)) AS t(id, foo, bar)";

    #[tokio::test]
    async fn with_column_to_json() -> Result<()> {
        let df = sample_frame()?.with_column(
            "json",
            to_json(named_struct(vec![
                lit("foo"),
                col("foo"),
                lit("bar"),
                col("bar"),
            ])),
        )?;

        // check that the new column is a nullable string
        let json_field = df.schema().field_with_unqualified_name("json")?;
        assert_eq!(json_field.data_type(), &DataType::Utf8);

        // check that new column added; the NULL `bar` is omitted
        let df_results = df.clone().collect().await?;
        assert_batches_sorted_eq!(
            [
                "+----+-----+-----+-----------------------+",
                "| id | foo | bar | json                  |",
                "+----+-----+-----+-----------------------+",
                "| 10 | 1   |     | {\"foo\":1}             |",
                "| 20 | 2   | bak | {\"foo\":2,\"bar\":\"bak\"} |",
                "| 30 | 3   | baz | {\"foo\":3,\"bar\":\"baz\"} |",
                "+----+-----+-----+-----------------------+",
            ],
            &df_results
        );

        // check that a column with the same name is overwritten, which folds the
        // source columns into a single JSON column
        let df_results_overwrite = df
            .clone()
            .with_column(
                "foo",
                to_json(named_struct(vec![
                    lit("foo"),
                    col("foo"),
                    lit("bar"),
                    col("bar"),
                ])),
            )?
            .collect()
            .await?;
        assert_batches_sorted_eq!(
            [
                "+----+-----------------------+-----+-----------------------+",
                "| id | foo                   | bar | json                  |",
                "+----+-----------------------+-----+-----------------------+",
                "| 10 | {\"foo\":1}             |     | {\"foo\":1}             |",
                "| 20 | {\"foo\":2,\"bar\":\"bak\"} | bak | {\"foo\":2,\"bar\":\"bak\"} |",
                "| 30 | {\"foo\":3,\"bar\":\"baz\"} | baz | {\"foo\":3,\"bar\":\"baz\"} |",
                "+----+-----------------------+-----+-----------------------+",
            ],
            &df_results_overwrite
        );

        // check that an existing struct column can be serialized with col(..),
        // with the same result as serializing the struct expression directly
        let df_struct_col = df
            .clone()
            .with_column(
                "s",
                named_struct(vec![lit("foo"), col("foo"), lit("bar"), col("bar")]),
            )?
            .with_column("json_from_struct", to_json(col("s")))?
            .select(vec![
                col("id"),
                col("foo"),
                col("bar"),
                col("json_from_struct"),
            ])?
            .with_column_renamed("json_from_struct", "json")?;
        assert_eq!(
            batches_to_sort_string(&df_struct_col.collect().await?),
            batches_to_sort_string(&df_results)
        );

        // check that a non-struct argument is rejected when the plan is built
        let err = df
            .with_column("bad", to_json(col("foo")))
            .unwrap_err()
            .to_string();
        assert_contains!(err, "to_json expects a struct argument");

        // check that the same result is produced through SQL: to_json is a
        // regular scalar UDF, so once registered it works on any struct
        // expression (`with_spark_features()` registers it together with all
        // other Spark functions; here only this one is registered)
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());

        let sql_results = ctx
            .sql(&format!(
                "SELECT id, foo, bar, to_json(named_struct('foo', foo, 'bar', bar)) AS json \
                 FROM {SAMPLE_TABLE_SQL}"
            ))
            .await?
            .collect()
            .await?;
        assert_eq!(
            batches_to_sort_string(&sql_results),
            batches_to_sort_string(&df_results)
        );

        // the SQL path rejects a non-struct argument at plan time as well
        let err = ctx
            .sql(&format!("SELECT to_json(foo) FROM {SAMPLE_TABLE_SQL}"))
            .await
            .unwrap_err()
            .to_string();
        assert_contains!(err, "to_json expects a struct argument");

        Ok(())
    }

    #[tokio::test]
    async fn with_column_to_json_edge_cases() -> Result<()> {
        // NULL fields are omitted from the object (Spark ignoreNullFields
        // default), so a row whose fields are all NULL becomes an empty object
        let df = dataframe!(
            "a" => [Some(1_i32), None],
            "b" => [None, None::<&str>],
        )?
        .with_column(
            "json",
            to_json(named_struct(vec![lit("a"), col("a"), lit("b"), col("b")])),
        )?;
        assert_batches_sorted_eq!(
            [
                "+---+---+---------+",
                "| a | b | json    |",
                "+---+---+---------+",
                "|   |   | {}      |",
                "| 1 |   | {\"a\":1} |",
                "+---+---+---------+",
            ],
            &df.collect().await?
        );

        // a constant struct is evaluated once at plan time (scalar path) and
        // broadcast to every row
        let df = dataframe!("id" => [1_i32, 2])?
            .with_column("json", to_json(named_struct(vec![lit("a"), lit(1)])))?;
        assert_batches_sorted_eq!(
            [
                "+----+---------+",
                "| id | json    |",
                "+----+---------+",
                "| 1  | {\"a\":1} |",
                "| 2  | {\"a\":1} |",
                "+----+---------+",
            ],
            &df.collect().await?
        );
        // nested structs and lists are serialized recursively
        let df = dataframe!("a" => [1_i32, 2])?.with_column(
            "json",
            to_json(named_struct(vec![
                lit("n"),
                named_struct(vec![lit("x"), col("a")]),
                lit("l"),
                make_array(vec![col("a"), lit(9)]),
            ])),
        )?;
        assert_batches_sorted_eq!(
            [
                "+---+-------------------------+",
                "| a | json                    |",
                "+---+-------------------------+",
                "| 1 | {\"n\":{\"x\":1},\"l\":[1,9]} |",
                "| 2 | {\"n\":{\"x\":2},\"l\":[2,9]} |",
                "+---+-------------------------+",
            ],
            &df.collect().await?
        );

        // string values are JSON-escaped; non-ASCII is emitted as-is
        let df = dataframe!("s" => [r#"he said "hi""#, r"a\b", "π"])?
            .with_column("json", to_json(named_struct(vec![lit("s"), col("s")])))?;
        assert_batches_sorted_eq!(
            [
                "+--------------+------------------------+",
                "| s            | json                   |",
                "+--------------+------------------------+",
                "| a\\b          | {\"s\":\"a\\\\b\"}           |",
                "| he said \"hi\" | {\"s\":\"he said \\\"hi\\\"\"} |",
                "| π            | {\"s\":\"π\"}              |",
                "+--------------+------------------------+",
            ],
            &df.collect().await?
        );

        // an empty input produces no rows but still carries the new column
        let df = sample_frame()?
            .filter(col("foo").gt(lit(100)))?
            .with_column("json", to_json(named_struct(vec![lit("foo"), col("foo")])))?;
        assert_eq!(
            df.schema().field_with_unqualified_name("json")?.data_type(),
            &DataType::Utf8
        );
        assert_eq!(df.count().await?, 0);

        Ok(())
    }

    /// A context with to_json and the nested functions used to build test
    /// values (`map`, `make_array`, ...; the `datafusion` dev-dependency is
    /// built without the `nested_expressions` feature).
    fn to_json_ctx() -> SessionContext {
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());
        for udf in datafusion_functions_nested::all_default_nested_functions() {
            ctx.register_udf((*udf).clone());
        }
        ctx
    }

    /// Evaluates `SELECT to_json(<expr>)` for each `(expr, expected)` pair and
    /// checks the JSON string.
    async fn assert_sql_json(cases: &[(&str, &str)]) -> Result<()> {
        let ctx = to_json_ctx();
        for (expr, expected) in cases {
            let batches = ctx
                .sql(&format!("SELECT to_json({expr}) AS j"))
                .await?
                .collect()
                .await?;
            let j = as_string_array(batches[0].column(0))?;
            assert_eq!(j.value(0), *expected, "expression: {expr}");
        }
        Ok(())
    }

    /// Asserts that planning `to_json(expr)` fails with a message containing `needle`.
    async fn assert_sql_plan_error(expr: &str, needle: &str) {
        let err = to_json_ctx()
            .sql(&format!("SELECT to_json({expr})"))
            .await
            .unwrap_err()
            .to_string();
        assert_contains!(err, needle);
    }

    #[tokio::test]
    async fn to_json_rejects_types_without_spark_json_text_sql() {
        assert_sql_plan_error(
            "named_struct('i', INTERVAL '1' DAY)",
            "field i has type Interval(MonthDayNano), which has no Spark JSON representation",
        )
        .await;
        assert_sql_plan_error(
            "named_struct('s', named_struct('l', make_array(INTERVAL '1' DAY)))",
            "field s.l[] has type Interval(MonthDayNano)",
        )
        .await;
        assert_sql_plan_error(
            "named_struct('m', map(make_array(make_array(1)), make_array(1)))",
            "field m has map key type List(",
        )
        .await;
    }

    #[tokio::test]
    async fn to_json_dates_and_dictionaries_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('d', DATE '2024-01-01')",
                r#"{"d":"2024-01-01"}"#,
            ),
            (
                "named_struct('d', arrow_cast(DATE '2024-01-01', 'Date64'))",
                r#"{"d":"2024-01-01"}"#,
            ),
            // dictionary-encoded values are written like their values
            (
                "named_struct('s', arrow_cast('a', 'Dictionary(Int32, Utf8)'))",
                r#"{"s":"a"}"#,
            ),
            (
                "named_struct('f', arrow_cast(1e10, 'Dictionary(Int32, Float64)'))",
                r#"{"f":1.0E10}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_keeps_null_map_values_sql() -> Result<()> {
        assert_sql_json(&[
            // NULL map value is kept
            (
                "named_struct('m', map(make_array('k'), make_array(CAST(NULL AS INT))))",
                r#"{"m":{"k":null}}"#,
            ),
            // nested map inside a map value
            (
                "named_struct('m', map(make_array('o'), make_array(map(make_array('k'), make_array(CAST(NULL AS INT))))))",
                r#"{"m":{"o":{"k":null}}}"#,
            ),
            // a struct inside a map value still omits its NULL fields
            (
                "named_struct('m', map(make_array('k'), make_array(named_struct('a', 1, 'b', CAST(NULL AS INT)))))",
                r#"{"m":{"k":{"a":1}}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_encodes_binary_as_base64_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('b', arrow_cast('abc', 'Binary'))",
                r#"{"b":"YWJj"}"#,
            ),
            (
                "named_struct('b', arrow_cast('abc', 'LargeBinary'))",
                r#"{"b":"YWJj"}"#,
            ),
            (
                "named_struct('b', arrow_cast('abc', 'BinaryView'))",
                r#"{"b":"YWJj"}"#,
            ),
            // nested in a list, in a struct and in a map value
            (
                "named_struct('l', make_array(arrow_cast('abc', 'Binary')))",
                r#"{"l":["YWJj"]}"#,
            ),
            (
                "named_struct('n', named_struct('b', arrow_cast('abc', 'Binary')))",
                r#"{"n":{"b":"YWJj"}}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(arrow_cast('abc', 'Binary'))))",
                r#"{"m":{"k":"YWJj"}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_non_finite_floats_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('x', CAST('NaN' AS DOUBLE))",
                r#"{"x":"NaN"}"#,
            ),
            (
                "named_struct('x', CAST('Infinity' AS DOUBLE))",
                r#"{"x":"Infinity"}"#,
            ),
            (
                "named_struct('x', CAST('-Infinity' AS DOUBLE))",
                r#"{"x":"-Infinity"}"#,
            ),
            (
                "named_struct('x', arrow_cast('NaN', 'Float32'))",
                r#"{"x":"NaN"}"#,
            ),
            // finite values are unchanged
            ("named_struct('x', 1.5)", r#"{"x":1.5}"#),
            // nested in a list, a struct and a map value
            (
                "named_struct('l', make_array(CAST('NaN' AS DOUBLE), 1.5))",
                r#"{"l":["NaN",1.5]}"#,
            ),
            (
                "named_struct('n', named_struct('x', CAST('Infinity' AS DOUBLE)))",
                r#"{"n":{"x":"Infinity"}}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(CAST('NaN' AS DOUBLE))))",
                r#"{"m":{"k":"NaN"}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_stringifies_map_keys_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('m', map(make_array(1), make_array('v')))",
                r#"{"m":{"1":"v"}}"#,
            ),
            (
                "named_struct('m', map(make_array(true), make_array('v')))",
                r#"{"m":{"true":"v"}}"#,
            ),
            (
                "named_struct('m', map(make_array(1.5), make_array('v')))",
                r#"{"m":{"1.5":"v"}}"#,
            ),
            // string keys are unchanged
            (
                "named_struct('m', map(make_array('k'), make_array('v')))",
                r#"{"m":{"k":"v"}}"#,
            ),
            // Spark's internal values: days, microseconds, Java's Double.toString
            // and BigDecimal.toString
            (
                "named_struct('m', map(make_array(DATE '2024-01-01'), make_array(1)))",
                r#"{"m":{"19723":1}}"#,
            ),
            (
                "named_struct('m', map(make_array(TIMESTAMP '2024-01-01 01:02:03.123456'), make_array(1)))",
                r#"{"m":{"1704070923123456":1}}"#,
            ),
            (
                "named_struct('m', map(make_array(1e10), make_array(1)))",
                r#"{"m":{"1.0E10":1}}"#,
            ),
            (
                "named_struct('m', map(make_array(CAST('NaN' AS DOUBLE)), make_array(1)))",
                r#"{"m":{"NaN":1}}"#,
            ),
            (
                "named_struct('m', map(make_array(CAST(0.0000001 AS DECIMAL(10, 8))), make_array(1)))",
                r#"{"m":{"1.0E-7":1}}"#,
            ),
            // recursively: an int-keyed map inside a list, a struct and a map value
            (
                "named_struct('l', make_array(map(make_array(1), make_array('v'))))",
                r#"{"l":[{"1":"v"}]}"#,
            ),
            (
                "named_struct('n', named_struct('m', map(make_array(1), make_array('v'))))",
                r#"{"n":{"m":{"1":"v"}}}"#,
            ),
            (
                "named_struct('m', map(make_array('o'), make_array(map(make_array(2), make_array('v')))))",
                r#"{"m":{"o":{"2":"v"}}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_timestamps_sql() -> Result<()> {
        assert_sql_json(&[
            // TIMESTAMP (instant): milliseconds, session time zone offset (`Z` for UTC)
            (
                "named_struct('t', arrow_cast('2024-01-01T01:02:03.123456Z', 'Timestamp(Microsecond, Some(\"UTC\"))'))",
                r#"{"t":"2024-01-01T01:02:03.123Z"}"#,
            ),
            // TIMESTAMP_NTZ: no offset, no conversion
            (
                "named_struct('t', TIMESTAMP '2024-01-01 01:02:03.123456')",
                r#"{"t":"2024-01-01T01:02:03.123"}"#,
            ),
        ])
        .await?;

        // the session time zone decides how an instant is written
        let ctx = SessionContext::new();
        ctx.register_udf((*crate::function::json::to_json()).clone());
        ctx.sql("SET datafusion.execution.time_zone = '+02:00'")
            .await?
            .collect()
            .await?;
        let batches = ctx
            .sql(
                "SELECT to_json(named_struct('t', arrow_cast('2024-01-01T01:02:03.123456Z', \
                 'Timestamp(Microsecond, Some(\"UTC\"))'))) AS j",
            )
            .await?
            .collect()
            .await?;
        let j = as_string_array(batches[0].column(0))?;
        assert_eq!(j.value(0), r#"{"t":"2024-01-01T03:02:03.123+02:00"}"#);
        Ok(())
    }

    #[tokio::test]
    async fn to_json_floats_use_java_to_string_sql() -> Result<()> {
        assert_sql_json(&[
            ("named_struct('x', 1e-5)", r#"{"x":1.0E-5}"#),
            (
                "named_struct('x', CAST(123456789 AS DOUBLE))",
                r#"{"x":1.23456789E8}"#,
            ),
            ("named_struct('x', 1e10)", r#"{"x":1.0E10}"#),
            ("named_struct('x', 1.0)", r#"{"x":1.0}"#),
            ("named_struct('x', 0.001)", r#"{"x":0.001}"#),
            (
                "named_struct('x', arrow_cast(0.1, 'Float32'))",
                r#"{"x":0.1}"#,
            ),
            // nested in a list and a map value
            (
                "named_struct('l', make_array(1e-5, 1.0))",
                r#"{"l":[1.0E-5,1.0]}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(1e10)))",
                r#"{"m":{"k":1.0E10}}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_escapes_strings_like_jackson_sql() -> Result<()> {
        assert_sql_json(&[
            (
                "named_struct('s', 'a' || chr(31) || 'b')",
                r#"{"s":"a\u001Fb"}"#,
            ),
            (
                "named_struct('s', 'q\"' || chr(10) || 'é')",
                r#"{"s":"q\"\né"}"#,
            ),
            (
                "named_struct('m', map(make_array(chr(31)), make_array(chr(1))))",
                r#"{"m":{"\u001F":"\u0001"}}"#,
            ),
            (
                "named_struct('l', make_array(chr(27), 'x'))",
                r#"{"l":["\u001B","x"]}"#,
            ),
        ])
        .await
    }

    #[tokio::test]
    async fn to_json_decimals_use_java_big_decimal_sql() -> Result<()> {
        assert_sql_json(&[
            ("named_struct('d', CAST(0 AS DECIMAL(38, 10)))", r#"{"d":0E-10}"#),
            (
                "named_struct('d', CAST(0.0000001 AS DECIMAL(10, 8)))",
                r#"{"d":1.0E-7}"#,
            ),
            ("named_struct('d', CAST(1.5 AS DECIMAL(10, 2)))", r#"{"d":1.50}"#),
            ("named_struct('d', CAST(123 AS DECIMAL(10, 0)))", r#"{"d":123}"#),
            (
                "named_struct('d', CAST(-0.05 AS DECIMAL(10, 2)))",
                r#"{"d":-0.05}"#,
            ),
            // nested in a list and a map value
            (
                "named_struct('l', make_array(CAST(0 AS DECIMAL(38, 10))))",
                r#"{"l":[0E-10]}"#,
            ),
            (
                "named_struct('m', map(make_array('k'), make_array(CAST(1.5 AS DECIMAL(10, 2)))))",
                r#"{"m":{"k":1.50}}"#,
            ),
        ])
        .await
    }
}
