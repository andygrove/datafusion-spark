// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, types::ArrowDictionaryKeyType};
use arrow::buffer::{Buffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, FieldRef};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use super::utils::hash_dictionary_values;
use crate::create_hashes_internal;

const DEFAULT_SEED: u32 = 42;

/// Spark-compatible `hash` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#hash>
#[user_doc(
    doc_section(label = "Hashing Functions"),
    description = "Returns the 32-bit Murmur3 hash (seed 42) of the arguments as an INT, matching Spark's `hash`. \
        DataFusion has no built-in equivalent. Arguments are hashed in order, and each argument's hash seeds the next, \
        so `hash(a, b)` equals `hash(named_struct('a', a, 'b', b))` and argument order matters. \
        NULL arguments, NULL list elements and NULL struct fields are skipped, so the result is never NULL: \
        `hash(NULL)` returns 42. The result depends on the input type, as in Spark: INT hashes 4 bytes and BIGINT 8, \
        and integer literals are BIGINT in DataFusion but INT in Spark, so cast them to INT to match Spark's results. \
        `-0.0` hashes like `0.0`. Decimals with precision up to 18 hash as their unscaled long, and wider decimals hash \
        the minimal big-endian two's complement bytes of the unscaled value. Lists hash each element in order \
        (nested lists flatten into the same chain, and an empty list leaves the seed unchanged), structs hash each field, \
        and maps hash each key then its value. Spark 4 rejects maps unless `spark.sql.legacy.allowHashOnMapType` is set. \
        Dictionary-encoded values hash like their decoded values. At least one argument is required.",
    syntax_example = "hash(expr1[, ..., exprN])",
    sql_example = r#"```sql
> SELECT hash('Spark', [CAST(123 AS INT)], CAST(2 AS INT));
-1321691492
> SELECT hash(CAST(1 AS INT));
-559580957
> SELECT hash(1);
-1712319331
> SELECT hash(NULL);
42
```"#,
    argument(
        name = "expr1",
        description = "Expression of any type to hash. NULL is skipped."
    ),
    argument(
        name = "exprN",
        description = "Further expressions, hashed in order with the previous hash as the seed."
    )
)]
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

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // When every argument is a scalar the result is the same for every row,
        // so hash a single row and return a scalar. Otherwise expand scalars to
        // the batch length so every array has `number_rows` entries.
        let all_scalars = args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));
        let num_rows = if all_scalars { 1 } else { args.number_rows };
        let arrays = args
            .args
            .iter()
            .map(|arg| arg.to_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        let mut hashes: Vec<u32> = vec![DEFAULT_SEED; num_rows];
        create_murmur3_hashes(&arrays, &mut hashes)?;

        if all_scalars {
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
    let (words, tail) = data.as_chunks::<4>();
    let mut h1 = seed;
    for word in words {
        h1 = mix_h1(h1, mix_k1(u32::from_le_bytes(*word)));
    }
    for &byte in tail {
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
    hash_dictionary_values::<K, u32>(
        array,
        hashes_buffer,
        seeds_are_pristine,
        create_murmur3_hashes,
    )
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
    use arrow::array::{DictionaryArray, StringArray};
    use arrow::datatypes::Int32Type;
    use datafusion_common::config::ConfigOptions;

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

    #[test]
    fn test_hash_dictionary_after_other_column_matches_decoded() {
        // The running hashes are no longer uniform once a first column is hashed,
        // so the dictionary must be expanded rather than hashed per value.
        let first: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let dict: ArrayRef = Arc::new(
            vec!["a", "b", "a"]
                .into_iter()
                .collect::<DictionaryArray<Int32Type>>(),
        );
        let decoded: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "a"]));
        let mut with_dict = vec![DEFAULT_SEED; 3];
        create_murmur3_hashes(&[Arc::clone(&first), dict], &mut with_dict).unwrap();
        let mut with_decoded = vec![DEFAULT_SEED; 3];
        create_murmur3_hashes(&[first, decoded], &mut with_decoded).unwrap();
        assert_eq!(with_dict, with_decoded);
    }

    #[test]
    fn test_hash_dictionary_matches_decoded() {
        let dict: DictionaryArray<Int32Type> =
            vec!["hello", "world", "hello"].into_iter().collect();
        let decoded: ArrayRef =
            Arc::new(StringArray::from(vec!["hello", "world", "hello"]));
        assert_eq!(hash_of(Arc::new(dict)), hash_of(decoded));
    }

    fn invoke(args: Vec<ColumnarValue>, number_rows: usize) -> Result<ColumnarValue> {
        let arg_fields = args
            .iter()
            .enumerate()
            .map(|(i, arg)| Arc::new(Field::new(format!("a{i}"), arg.data_type(), true)))
            .collect();
        SparkHash::new().invoke_with_args(ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows,
            return_field: Arc::new(Field::new("hash", DataType::Int32, false)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    // An all-literal hash(...) that is not constant-folded (e.g. a physical
    // expression built directly) is evaluated with the batch's row count, while
    // every argument is a scalar. This cannot be reached from SQL.
    #[test]
    fn test_hash_all_scalar_args_with_many_rows() -> Result<()> {
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("hello".to_string()))),
        ];
        let mut expected = vec![DEFAULT_SEED];
        create_murmur3_hashes(
            &[
                Arc::new(Int32Array::from(vec![1])),
                Arc::new(StringArray::from(vec!["hello"])),
            ],
            &mut expected,
        )?;
        match invoke(args, 3)? {
            ColumnarValue::Scalar(ScalarValue::Int32(Some(h))) => {
                assert_eq!(h, expected[0] as i32)
            }
            other => panic!("expected an Int32 scalar, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn test_hash_scalar_and_array_args() -> Result<()> {
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["a", "b", "c"]))),
        ];
        let mut expected = vec![DEFAULT_SEED; 3];
        create_murmur3_hashes(
            &[
                Arc::new(Int32Array::from(vec![1, 1, 1])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
            &mut expected,
        )?;
        let ColumnarValue::Array(out) = invoke(args, 3)? else {
            panic!("expected an array");
        };
        let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
        let expected: Vec<i32> = expected.into_iter().map(|h| h as i32).collect();
        assert_eq!(out.values().to_vec(), expected);
        Ok(())
    }
}
