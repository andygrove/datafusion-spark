// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int32Array, types::ArrowDictionaryKeyType,
};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::compute::take;
use arrow::datatypes::{ArrowNativeType, DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    Volatility,
};

use crate::create_hashes_internal;

const DEFAULT_SEED: u32 = 42;

/// Spark-compatible `hash` function (32-bit Murmur3, seed 42).
/// <https://spark.apache.org/docs/latest/api/sql/index.html#hash>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkHash {
    signature: Signature,
}

impl Default for SparkHash {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkHash {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkHash {
    fn name(&self) -> &str {
        "hash"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, _args: ReturnFieldArgs) -> Result<FieldRef> {
        // Spark's HashExpression overrides nullable to false: NULL inputs are
        // skipped and the seed is used, so the result is never null.
        Ok(Arc::new(Field::new(self.name(), DataType::Int32, false)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let mut hashes: Vec<u32> = vec![DEFAULT_SEED; num_rows];

        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        create_murmur3_hashes(&arrays, &mut hashes)?;

        if num_rows == 1 {
            Ok(ColumnarValue::Scalar(ScalarValue::Int32(Some(
                hashes[0] as i32,
            ))))
        } else {
            // Reinterpret Vec<u32> as ScalarBuffer<i32> without copying.
            let buffer = ScalarBuffer::<i32>::from(Buffer::from_vec(hashes));
            Ok(ColumnarValue::Array(Arc::new(Int32Array::new(
                buffer, None,
            ))))
        }
    }
}

#[inline]
fn mix_k1(k1: u32) -> u32 {
    k1.wrapping_mul(0xcc9e2d51)
        .rotate_left(15)
        .wrapping_mul(0x1b873593)
}

#[inline]
fn mix_h1(h1: u32, k1: u32) -> u32 {
    (h1 ^ k1)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe6546b64)
}

#[inline]
fn fmix(mut h1: u32, len: u32) -> u32 {
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85ebca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2ae35);
    h1 ^= h1 >> 16;
    h1
}

/// Spark's `Murmur3_x86_32.hashUnsafeBytes`: whole 4-byte little-endian words
/// first, then each remaining tail byte is mixed separately as a sign-extended
/// word (this differs from the canonical MurmurHash3 tail handling).
#[inline]
fn spark_compatible_murmur3_hash<T: AsRef<[u8]>>(data: T, seed: u32) -> u32 {
    let data = data.as_ref();
    let mut chunks = data.chunks_exact(4);
    let mut h1 = seed;
    for chunk in &mut chunks {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        h1 = mix_h1(h1, mix_k1(word));
    }
    for &byte in chunks.remainder() {
        h1 = mix_h1(h1, mix_k1(byte as i8 as i32 as u32));
    }
    fmix(h1, data.len() as u32)
}

/// Hash the values in a dictionary array using murmur3.
fn create_murmur3_hashes_dictionary<K: ArrowDictionaryKeyType>(
    array: &ArrayRef,
    hashes_buffer: &mut [u32],
    seeds_are_pristine: bool,
) -> Result<()> {
    let dict_array = array.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();
    if !seeds_are_pristine {
        let unpacked = take(dict_array.values().as_ref(), dict_array.keys(), None)?;
        create_murmur3_hashes(&[unpacked], hashes_buffer)?;
    } else {
        // Hash each dictionary value once, then look up by key.
        let dict_values = Arc::clone(dict_array.values());
        // Every row has the same running hash here, so start from it instead of 42
        let seed = hashes_buffer.first().copied().unwrap_or(DEFAULT_SEED);
        let mut dict_hashes = vec![seed; dict_values.len()];
        create_murmur3_hashes(&[dict_values], &mut dict_hashes)?;

        for (hash, key) in hashes_buffer.iter_mut().zip(dict_array.keys().iter()) {
            if let Some(key) = key {
                *hash = dict_hashes[key.as_usize()]
            }
            // No update for Null keys, consistent with other types.
        }
    }
    Ok(())
}

/// Create murmur3 hash values for every row, based on the values in the columns.
///
/// The number of rows to hash is determined by `hashes_buffer.len()`.
/// `hashes_buffer` should be pre-sized appropriately and seeded with the
/// initial hash value (Spark uses `42`).
fn create_murmur3_hashes(arrays: &[ArrayRef], hashes_buffer: &mut [u32]) -> Result<()> {
    create_hashes_internal!(
        arrays,
        hashes_buffer,
        spark_compatible_murmur3_hash,
        create_murmur3_hashes_dictionary,
        create_murmur3_hashes
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Float64Array, StringArray};
    use arrow::datatypes::Int32Type;

    fn hash_of(array: ArrayRef) -> Vec<i32> {
        let mut hashes = vec![DEFAULT_SEED; array.len()];
        create_murmur3_hashes(&[array], &mut hashes).unwrap();
        hashes.into_iter().map(|h| h as i32).collect()
    }

    #[test]
    fn test_hash_nullability() -> Result<()> {
        let func = SparkHash::new();
        let nullable: FieldRef = Arc::new(Field::new("a", DataType::Int32, true));
        let out = func.return_field_from_args(ReturnFieldArgs {
            arg_fields: &[nullable],
            scalar_arguments: &[None],
        })?;
        assert!(!out.is_nullable());
        assert_eq!(out.data_type(), &DataType::Int32);
        Ok(())
    }

    // Expected values are taken from Comet's Spark-verified murmur3 tests.
    #[test]
    fn test_hash_i32() {
        assert_eq!(
            hash_of(Arc::new(Int32Array::from(vec![1, 0, -1]))),
            vec![-559580957, 933211791, -1604776387]
        );
    }

    #[test]
    fn test_hash_str() {
        let array: ArrayRef = Arc::new(StringArray::from(vec![
            "hello", "", "a", "ab", "abc", "abcd", "abcde", "天地",
        ]));
        // Covers every tail length (0-3 bytes) and multi-byte UTF-8.
        assert_eq!(
            hash_of(array),
            [
                3286402344u32,
                142593372,
                1485273170,
                0xfa37157b,
                1322437556,
                0xe860e5cc,
                814637928,
                2395000894
            ]
            .map(|h| h as i32)
        );
    }

    #[test]
    fn test_hash_negative_zero() {
        let f32s: ArrayRef = Arc::new(Float32Array::from(vec![0.0f32, -0.0f32]));
        let h = hash_of(f32s);
        assert_eq!(h[0], h[1]);
        let f64s: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64, -0.0f64]));
        let h = hash_of(f64s);
        assert_eq!(h[0], h[1]);
    }

    #[test]
    fn test_hash_null_keeps_seed() {
        let array: ArrayRef = Arc::new(Int32Array::from(vec![None, Some(1)]));
        assert_eq!(hash_of(array)[0], 42);
    }

    #[test]
    fn test_hash_dictionary_matches_decoded() {
        let dict: DictionaryArray<Int32Type> =
            vec!["hello", "world", "hello"].into_iter().collect();
        let decoded: ArrayRef =
            Arc::new(StringArray::from(vec!["hello", "world", "hello"]));
        assert_eq!(hash_of(Arc::new(dict)), hash_of(decoded));
    }
}
