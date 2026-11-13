// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, ArrowPrimitiveType, AsArray, Int32Array, PrimitiveArray};
use arrow::compute;
use arrow::datatypes::{
    ArrowNativeType, DataType, Field, FieldRef, Int32Type, Int64Type, UInt32Type,
    UInt64Type,
};
use datafusion_common::types::{
    NativeType, logical_int8, logical_int16, logical_int32, logical_int64, logical_uint8,
    logical_uint16, logical_uint32, logical_uint64,
};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, internal_err};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs,
    ScalarUDFImpl, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_functions::utils::make_scalar_function;
use datafusion_macros::user_doc;

/// Bitwise left shift on elements in `value` by corresponding `shift` amount.
/// The shift amount is normalized to the bit width of the type, matching Spark/Java
/// semantics for negative and large shifts.
fn shift_left<T>(
    value: &PrimitiveArray<T>,
    shift: &Int32Array,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: std::ops::Shl<i32, Output = T::Native>,
{
    let bit_num = (T::Native::get_byte_width() * 8) as i32;
    let result = compute::binary::<_, Int32Type, _, _>(
        value,
        shift,
        |value: T::Native, shift: i32| {
            let shift = ((shift % bit_num) + bit_num) % bit_num;
            value << shift
        },
    )?;
    Ok(result)
}

/// Bitwise right shift on elements in `value` by corresponding `shift` amount.
/// The shift amount is normalized to the bit width of the type, matching Spark/Java
/// semantics for negative and large shifts.
fn shift_right<T>(
    value: &PrimitiveArray<T>,
    shift: &Int32Array,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: std::ops::Shr<i32, Output = T::Native>,
{
    let bit_num = (T::Native::get_byte_width() * 8) as i32;
    let result = compute::binary::<_, Int32Type, _, _>(
        value,
        shift,
        |value: T::Native, shift: i32| {
            let shift = ((shift % bit_num) + bit_num) % bit_num;
            value >> shift
        },
    )?;
    Ok(result)
}

/// Trait for performing an unsigned right shift (logical shift right).
/// This is used to mimic Java's `>>>` operator, which does not exist in Rust.
/// For unsigned types, this is just the normal right shift.
/// For signed types, this casts to the unsigned type, shifts, then casts back.
trait UShr {
    fn ushr(self, rhs: i32) -> Self;
}

impl UShr for u32 {
    fn ushr(self, rhs: i32) -> Self {
        self >> rhs
    }
}

impl UShr for u64 {
    fn ushr(self, rhs: i32) -> Self {
        self >> rhs
    }
}

impl UShr for i32 {
    fn ushr(self, rhs: i32) -> Self {
        ((self as u32) >> rhs) as i32
    }
}

impl UShr for i64 {
    fn ushr(self, rhs: i32) -> Self {
        ((self as u64) >> rhs) as i64
    }
}

/// Bitwise unsigned right shift on elements in `value` by corresponding `shift`
/// amount. The shift amount is normalized to the bit width of the type, matching
/// Spark/Java semantics for negative and large shifts.
fn shift_right_unsigned<T>(
    value: &PrimitiveArray<T>,
    shift: &Int32Array,
) -> Result<PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
    T::Native: UShr,
{
    let bit_num = (T::Native::get_byte_width() * 8) as i32;
    let result = compute::binary::<_, Int32Type, _, _>(
        value,
        shift,
        |value: T::Native, shift: i32| {
            let shift = ((shift % bit_num) + bit_num) % bit_num;
            value.ushr(shift)
        },
    )?;
    Ok(result)
}

fn shift_inner(
    arrays: &[ArrayRef],
    name: &str,
    bit_shift_type: BitShiftType,
) -> Result<ArrayRef> {
    let [value_array, shift_array] = take_function_args(name, arrays)?;
    let shift_array = shift_array.as_primitive::<Int32Type>();

    fn shift<T>(
        value: &PrimitiveArray<T>,
        shift: &Int32Array,
        bit_shift_type: BitShiftType,
    ) -> Result<PrimitiveArray<T>>
    where
        T: ArrowPrimitiveType,
        T::Native: std::ops::Shl<i32, Output = T::Native>
            + std::ops::Shr<i32, Output = T::Native>
            + UShr,
    {
        match bit_shift_type {
            BitShiftType::Left => shift_left(value, shift),
            BitShiftType::Right => shift_right(value, shift),
            BitShiftType::RightUnsigned => shift_right_unsigned(value, shift),
        }
    }

    match value_array.data_type() {
        DataType::Int32 => {
            let value_array = value_array.as_primitive::<Int32Type>();
            Ok(Arc::new(shift(value_array, shift_array, bit_shift_type)?))
        }
        DataType::Int64 => {
            let value_array = value_array.as_primitive::<Int64Type>();
            Ok(Arc::new(shift(value_array, shift_array, bit_shift_type)?))
        }
        DataType::UInt32 => {
            let value_array = value_array.as_primitive::<UInt32Type>();
            Ok(Arc::new(shift(value_array, shift_array, bit_shift_type)?))
        }
        DataType::UInt64 => {
            let value_array = value_array.as_primitive::<UInt64Type>();
            Ok(Arc::new(shift(value_array, shift_array, bit_shift_type)?))
        }
        dt => {
            internal_err!("{name} function does not support data type: {dt}")
        }
    }
}

#[derive(Debug, Hash, Copy, Clone, Eq, PartialEq)]
enum BitShiftType {
    Left,
    Right,
    RightUnsigned,
}

/// Spark-compatible `shiftleft`, `shiftright` and `shiftrightunsigned` expressions.
///
/// One struct implements all three functions, so each function's documentation is
/// attached to its own marker type below.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#shiftleft>
#[derive(Debug, Hash, Eq, PartialEq)]
pub struct SparkBitShift {
    signature: Signature,
    name: &'static str,
    bit_shift_type: BitShiftType,
}

impl SparkBitShift {
    fn new(name: &'static str, bit_shift_type: BitShiftType) -> Self {
        let shift_amount = Coercion::new_implicit(
            TypeSignatureClass::Native(logical_int32()),
            vec![TypeSignatureClass::Integer],
            NativeType::Int32,
        );
        Self {
            signature: Signature::one_of(
                vec![
                    // Upcast small ints to 32bit
                    TypeSignature::Coercible(vec![
                        Coercion::new_implicit(
                            TypeSignatureClass::Native(logical_int32()),
                            vec![
                                TypeSignatureClass::Native(logical_int8()),
                                TypeSignatureClass::Native(logical_int16()),
                            ],
                            NativeType::Int32,
                        ),
                        shift_amount.clone(),
                    ]),
                    TypeSignature::Coercible(vec![
                        Coercion::new_implicit(
                            TypeSignatureClass::Native(logical_uint32()),
                            vec![
                                TypeSignatureClass::Native(logical_uint8()),
                                TypeSignatureClass::Native(logical_uint16()),
                            ],
                            NativeType::UInt32,
                        ),
                        shift_amount.clone(),
                    ]),
                    // Otherwise accept direct 64 bit integers
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_int64())),
                        shift_amount.clone(),
                    ]),
                    TypeSignature::Coercible(vec![
                        Coercion::new_exact(TypeSignatureClass::Native(logical_uint64())),
                        shift_amount.clone(),
                    ]),
                ],
                Volatility::Immutable,
            ),
            name,
            bit_shift_type,
        }
    }

    pub fn left() -> Self {
        Self::new("shiftleft", BitShiftType::Left)
    }

    pub fn right() -> Self {
        Self::new("shiftright", BitShiftType::Right)
    }

    pub fn right_unsigned() -> Self {
        Self::new("shiftrightunsigned", BitShiftType::RightUnsigned)
    }
}

#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Shifts the bits of `expr` left by `n` positions. Bits shifted past the top are lost, \
        so the result can change sign or wrap to 0. \
        As in Spark (and Java), the shift amount is taken modulo the bit width of `expr` (32 for INT, 64 for BIGINT), \
        so `shiftleft(1::int, 32)` is 1 and a negative `n` shifts by the bit width plus `n`. \
        TINYINT and SMALLINT are widened to INT, and the result has the type of the widened `expr`. \
        Returns NULL if either input is NULL. DataFusion has no built-in `shiftleft` function.",
    syntax_example = "shiftleft(expr, n)",
    sql_example = r#"```sql
> SELECT shiftleft(2::int, 1);
4
> SELECT shiftleft(2147483647::int, 1);
-2
> SELECT shiftleft(1::int, 32);
1
```"#,
    argument(
        name = "expr",
        description = "Integer expression to shift. NULL returns NULL."
    ),
    argument(
        name = "n",
        description = "Integer number of bit positions, taken modulo the bit width of `expr`. NULL returns NULL."
    )
)]
struct ShiftLeftDoc;

#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Shifts the bits of `expr` right by `n` positions, copying the sign bit into the vacated high bits \
        (an arithmetic shift, so negative values stay negative). \
        As in Spark (and Java), the shift amount is taken modulo the bit width of `expr` (32 for INT, 64 for BIGINT), \
        so `shiftright(1::int, 32)` is 1 and a negative `n` shifts by the bit width plus `n`. \
        TINYINT and SMALLINT are widened to INT, and the result has the type of the widened `expr`. \
        Returns NULL if either input is NULL. See `shiftrightunsigned` for a shift that fills with zeros. \
        DataFusion has no built-in `shiftright` function.",
    syntax_example = "shiftright(expr, n)",
    sql_example = r#"```sql
> SELECT shiftright(16::int, 3);
2
> SELECT shiftright(-16::int, 3);
-2
> SELECT shiftright(4::int, -1);
0
```"#,
    argument(
        name = "expr",
        description = "Integer expression to shift. NULL returns NULL."
    ),
    argument(
        name = "n",
        description = "Integer number of bit positions, taken modulo the bit width of `expr`. NULL returns NULL."
    )
)]
struct ShiftRightDoc;

#[user_doc(
    doc_section(label = "Math Functions"),
    description = "Shifts the bits of `expr` right by `n` positions, filling the vacated high bits with zeros \
        (a logical shift, Java's `>>>`), so a negative value becomes a large positive one. \
        As in Spark (and Java), the shift amount is taken modulo the bit width of `expr` (32 for INT, 64 for BIGINT), \
        so `shiftrightunsigned(1::int, 32)` is 1 and a negative `n` shifts by the bit width plus `n`. \
        TINYINT and SMALLINT are widened to INT, and the result has the type of the widened `expr`. \
        Returns NULL if either input is NULL. DataFusion has no built-in `shiftrightunsigned` function.",
    syntax_example = "shiftrightunsigned(expr, n)",
    sql_example = r#"```sql
> SELECT shiftrightunsigned(16::int, 3);
2
> SELECT shiftrightunsigned(-16::int, 3);
536870910
```"#,
    argument(
        name = "expr",
        description = "Integer expression to shift. NULL returns NULL."
    ),
    argument(
        name = "n",
        description = "Integer number of bit positions, taken modulo the bit width of `expr`. NULL returns NULL."
    )
)]
struct ShiftRightUnsignedDoc;

impl ScalarUDFImpl for SparkBitShift {
    fn name(&self) -> &str {
        self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());
        let data_type = args.arg_fields[0].data_type().clone();
        Ok(Arc::new(Field::new(self.name(), data_type, nullable)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        match self.bit_shift_type {
            BitShiftType::Left => ShiftLeftDoc.doc(),
            BitShiftType::Right => ShiftRightDoc.doc(),
            BitShiftType::RightUnsigned => ShiftRightUnsignedDoc.doc(),
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let inner = |arr: &[ArrayRef]| -> Result<ArrayRef> {
            shift_inner(arr, self.name(), self.bit_shift_type)
        };
        make_scalar_function(inner, vec![])(&args.args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_shift_nullability() -> Result<()> {
        let func = SparkBitShift::left();

        let non_nullable_value: FieldRef =
            Arc::new(Field::new("value", DataType::Int64, false));
        let non_nullable_shift: FieldRef =
            Arc::new(Field::new("shift", DataType::Int32, false));

        let out = func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[
                Arc::clone(&non_nullable_value),
                Arc::clone(&non_nullable_shift),
            ],
            scalar_arguments: &[None, None],
        })?;

        assert_eq!(out.data_type(), non_nullable_value.data_type());
        assert!(
            !out.is_nullable(),
            "shift result should be non-nullable when both inputs are non-nullable"
        );

        let nullable_value: FieldRef =
            Arc::new(Field::new("value", DataType::Int64, true));
        let out_nullable_value = func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[Arc::clone(&nullable_value), Arc::clone(&non_nullable_shift)],
            scalar_arguments: &[None, None],
        })?;
        assert!(
            out_nullable_value.is_nullable(),
            "shift result should be nullable when value is nullable"
        );

        let nullable_shift: FieldRef =
            Arc::new(Field::new("shift", DataType::Int32, true));
        let out_nullable_shift = func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[non_nullable_value, nullable_shift],
            scalar_arguments: &[None, None],
        })?;
        assert!(
            out_nullable_shift.is_nullable(),
            "shift result should be nullable when shift is nullable"
        );

        Ok(())
    }
}
