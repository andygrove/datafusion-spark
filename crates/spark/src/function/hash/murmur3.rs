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
        `-0.0` hashes like `0.0`, and every NaN hashes like the canonical NaN. Dates hash as INT days and \
        timestamps as BIGINT microseconds, whatever the Arrow unit (DataFusion's TIMESTAMP literals are \
        nanoseconds, floored to microseconds). TIME hashes as BIGINT nanoseconds since midnight, a year-month \
        interval (`Interval(YearMonth)`) as INT months and a day-time interval (`Duration(Microsecond)`) as BIGINT \
        microseconds. A calendar interval (`Interval(MonthDayNano)`) hashes its microseconds and then its months, \
        and leaves out its days, as Spark 4.2's generated code does (Spark 4.3 includes the days, see SPARK-58236). \
        DataFusion types `INTERVAL '1' DAY` as a calendar interval, so it hashes like a zero interval, where Spark \
        hashes it as a day-time interval. Use `arrow_cast(86400000000, 'Duration(Microsecond)')` to match Spark. Decimals with precision up to 18 hash as their unscaled long, and wider decimals hash \
        the minimal big-endian two's complement bytes of the unscaled value. Lists hash each element in order \
        (nested lists flatten into the same chain, and an empty list leaves the seed unchanged), structs hash each field, \
        and maps hash each key then its value. This crate hashes maps as if Spark's `spark.sql.legacy.allowHashOnMapType` \
        were set, where Spark 4 rejects them by default (see https://github.com/apache/datafusion-spark/issues/69). \
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

    /// Non-canonical NaNs (sign bit or payload set) hash like the canonical NaN,
    /// as in Spark, both at the top level and inside lists.
    #[test]
    fn test_hash_non_canonical_nan() {
        use arrow::array::{Float32Array, Float64Array, ListArray};
        use arrow::buffer::OffsetBuffer;

        let nans = vec![
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff8_dead_beef_0001),
        ];
        // Spark: hash(CAST('NaN' AS DOUBLE))
        assert_eq!(
            hash_of(Arc::new(Float64Array::from(nans.clone()))),
            vec![-1281358385; 4]
        );
        let list: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Float64, false)),
            OffsetBuffer::new(vec![0i32, 1, 2, 3, 4].into()),
            Arc::new(Float64Array::from(nans)),
            None,
        ));
        assert_eq!(hash_of(list), vec![-1281358385; 4]);

        let f32s: ArrayRef = Arc::new(Float32Array::from(vec![
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001),
        ]));
        // Spark: hash(CAST('NaN' AS FLOAT))
        assert_eq!(hash_of(f32s), vec![-349261430; 3]);
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

    /// Tests ported from Apache DataFusion Comet's
    /// `native/spark-expr/src/hash_funcs/murmur3.rs` (commit ba16e1a), with their expected
    /// values unchanged: Comet produced them and they match Spark. Comet's two tests of its
    /// gather-eligibility fast-path selection are not ported, since this crate has no such
    /// selection.
    mod comet {
        use std::sync::Arc;

        use arrow::array::{
            Array, ArrayRef, Float32Array, Float64Array, Int8Array, Int32Array,
            Int64Array, StringArray,
        };
        use datafusion_common::config::ConfigOptions;
        use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl};

        use super::super::create_murmur3_hashes;
        use crate::function::hash::xxhash64::SparkXxhash64;

        /// Produced by the per-element `hash_list_array!` implementation. An empty list and a null
        /// list both leave the seed untouched.
        const EXPECTED_LIST_OF_STRUCT: [u32; 6] =
            [262891156, 1206178823, 42, 42, 3798669693, 2032748937];

        /// Stands in for Comet's `test_hashes_with_nulls!`: hashes `values` from seed 42 and
        /// checks `expected`, then again with nulls inserted at the front and the middle, which
        /// must leave the seed untouched.
        fn test_murmur3_hash<I: Clone, T: Array + From<Vec<Option<I>>> + 'static>(
            values: Vec<Option<I>>,
            expected: Vec<u32>,
        ) {
            let len = values.len();
            let mut hashes = vec![42u32; len];
            create_murmur3_hashes(
                &[Arc::new(T::from(values.clone())) as ArrayRef],
                &mut hashes,
            )
            .unwrap();
            assert_eq!(hashes, expected);

            let median = len / 2;
            let mut values_with_nulls = values;
            let mut expected_with_nulls = expected;
            values_with_nulls.insert(0, None);
            values_with_nulls.insert(median, None);
            expected_with_nulls.insert(0, 42);
            expected_with_nulls.insert(median, 42);
            let mut hashes = vec![42u32; len + 2];
            create_murmur3_hashes(
                &[Arc::new(T::from(values_with_nulls)) as ArrayRef],
                &mut hashes,
            )
            .unwrap();
            assert_eq!(hashes, expected_with_nulls);
        }

        /// Stands in for Comet's `create_xxhash64_hashes`, which is private to the `xxhash64`
        /// module here, by evaluating the `xxhash64` UDF. That always starts from Spark's seed 42,
        /// which is the only seed these tests use.
        fn create_xxhash64_hashes(
            arrays: &[ArrayRef],
            hashes_buffer: &mut [u64],
        ) -> datafusion_common::Result<()> {
            assert!(
                hashes_buffer.iter().all(|&h| h == 42),
                "only seed 42 is supported"
            );
            let args: Vec<ColumnarValue> = arrays
                .iter()
                .map(|a| ColumnarValue::Array(Arc::clone(a)))
                .collect();
            let arg_fields = arrays
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    Arc::new(arrow::datatypes::Field::new(
                        format!("a{i}"),
                        a.data_type().clone(),
                        true,
                    ))
                })
                .collect();
            let out = SparkXxhash64::new().invoke_with_args(ScalarFunctionArgs {
                args,
                arg_fields,
                number_rows: hashes_buffer.len(),
                return_field: Arc::new(arrow::datatypes::Field::new(
                    "xxhash64",
                    arrow::datatypes::DataType::Int64,
                    false,
                )),
                config_options: Arc::new(ConfigOptions::default()),
            })?;
            let ColumnarValue::Array(out) = out else {
                panic!("expected an array");
            };
            let out = out.as_any().downcast_ref::<Int64Array>().unwrap();
            for (h, v) in hashes_buffer.iter_mut().zip(out.values()) {
                *h = *v as u64;
            }
            Ok(())
        }

        #[test]
        fn test_dictionary_element_in_list_matches_decoded() {
            use arrow::array::{DictionaryArray, Int32Array, ListArray};
            use arrow::buffer::OffsetBuffer;
            use arrow::datatypes::{Field, Int8Type};

            let values: ArrayRef = Arc::new(Int32Array::from(vec![10, 20]));
            let keys = Int8Array::from(vec![0i8, 1]);
            let dict: ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
            let decoded: ArrayRef = Arc::new(Int32Array::from(vec![10, 20]));

            // One row holding both elements, so the second element's hash chains onto the first.
            let as_list = |elems: ArrayRef| -> ArrayRef {
                Arc::new(ListArray::new(
                    Arc::new(Field::new("item", elems.data_type().clone(), true)),
                    OffsetBuffer::new(vec![0i32, 2].into()),
                    elems,
                    None,
                ))
            };

            let mut from_dict = vec![42u32; 1];
            create_murmur3_hashes(&[as_list(dict)], &mut from_dict).unwrap();
            let mut from_decoded = vec![42u32; 1];
            create_murmur3_hashes(&[as_list(decoded)], &mut from_decoded).unwrap();

            assert_eq!(
                from_dict, from_decoded,
                "a dictionary-encoded list element must hash like the decoded value"
            );
        }

        /// The fast path must survive for a genuine first column, including one with a caller-supplied
        /// seed that is not Spark's 42, since the test for it is that every row is seeded alike.
        #[test]
        fn test_top_level_dictionary_matches_decoded() {
            use arrow::array::{DictionaryArray, Int32Array};
            use arrow::datatypes::Int8Type;

            let values: ArrayRef = Arc::new(Int32Array::from(vec![10, 20]));
            let keys = Int8Array::from(vec![0i8, 1, 0]);
            let dict: ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
            let decoded: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 10]));

            for seed in [42u32, 7u32] {
                let mut a = vec![seed; 3];
                create_murmur3_hashes(&[Arc::clone(&dict)], &mut a).unwrap();
                let mut b = vec![seed; 3];
                create_murmur3_hashes(&[Arc::clone(&decoded)], &mut b).unwrap();
                assert_eq!(
                    a, b,
                    "top-level dictionary must match decoded for seed {seed}"
                );
            }
        }

        /// The uniformity check is what makes the fast path safe, and a single-row case cannot pin it:
        /// one row is trivially uniform. This hashes several rows whose incoming seeds all differ, so a
        /// dictionary first column has to take the unpacking fallback. It also covers a null key and a
        /// key pointing at a null dictionary value, since both skip the hash update.
        #[test]
        fn test_dictionary_with_nonuniform_seeds_matches_decoded() {
            use arrow::array::{DictionaryArray, Int32Array};
            use arrow::datatypes::Int8Type;

            // values[2] is null, and one key is itself null
            let values: ArrayRef =
                Arc::new(Int32Array::from(vec![Some(10), Some(20), None]));
            let keys = Int8Array::from(vec![Some(0), Some(1), Some(2), None, Some(0)]);
            let dict: ArrayRef =
                Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
            // The same logical data with the dictionary resolved.
            let decoded: ArrayRef = Arc::new(Int32Array::from(vec![
                Some(10),
                Some(20),
                None,
                None,
                Some(10),
            ]));

            let seeds: Vec<u32> = vec![7, 38, 69, 100, 131];
            let mut from_dict = seeds.clone();
            create_murmur3_hashes(&[dict], &mut from_dict).unwrap();
            let mut from_decoded = seeds;
            create_murmur3_hashes(&[decoded], &mut from_decoded).unwrap();

            assert_eq!(
                from_dict, from_decoded,
                "with per-row seeds a dictionary must hash like the decoded array"
            );
        }

        /// Arrow lets a `StructArray`'s children carry their own validity, so at a row where the struct
        /// itself is null the child buffer can still hold a value. Spark hashes a null struct as the
        /// seed, so those hidden child values must not reach the hash.
        #[test]
        fn test_null_struct_ignores_hidden_child_values() {
            use arrow::array::{Int32Array, StructArray};
            use arrow::buffer::NullBuffer;
            use arrow::datatypes::{DataType, Field, Fields};

            let fields: Fields =
                vec![Arc::new(Field::new("a", DataType::Int32, true))].into();
            // Row 1 is a null struct whose child still holds 999.
            let child: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), Some(999)]));
            let nulls = NullBuffer::from(vec![true, false]);
            let with_hidden: ArrayRef = Arc::new(StructArray::new(
                fields.clone(),
                vec![Arc::clone(&child)],
                Some(nulls.clone()),
            ));
            // Same shape, but the hidden slot is null too.
            let child_null: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None]));
            let without_hidden: ArrayRef =
                Arc::new(StructArray::new(fields, vec![child_null], Some(nulls)));

            let mut a = vec![42u32; 2];
            create_murmur3_hashes(&[with_hidden], &mut a).unwrap();
            let mut b = vec![42u32; 2];
            create_murmur3_hashes(&[without_hidden], &mut b).unwrap();

            assert_eq!(
                a, b,
                "a null struct must hash the same regardless of what its child buffer holds"
            );
            assert_eq!(a[1], 42, "a null struct must leave the seed untouched");
        }

        /// The struct branch is also reached once per element when hashing `array<struct<..>>`. The
        /// test above hashes a struct directly, so it does not cover that route.
        ///
        /// Here the null is the list *element* itself, with valid elements either side so the chaining
        /// is exercised.
        #[test]
        fn test_null_struct_element_of_list_ignores_hidden_child_values() {
            use arrow::array::{Int32Array, ListArray, StructArray};
            use arrow::buffer::{NullBuffer, OffsetBuffer};
            use arrow::datatypes::{DataType, Field, Fields};

            let fields: Fields =
                vec![Arc::new(Field::new("a", DataType::Int32, true))].into();
            // Element 1 is a null struct whose child still holds 999; elements 0 and 2 are valid.
            let element_nulls = NullBuffer::from(vec![true, false, true]);
            let with_hidden: ArrayRef = Arc::new(StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![Some(1), Some(999), Some(3)]))
                        as ArrayRef,
                ],
                Some(element_nulls.clone()),
            ));
            // The same shape with the hidden slot null as well.
            let without_hidden: ArrayRef = Arc::new(StructArray::new(
                fields,
                vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef],
                Some(element_nulls),
            ));

            // One row holding all three elements, so element hashes chain in order.
            let as_list = |elements: ArrayRef| -> ArrayRef {
                Arc::new(ListArray::new(
                    Arc::new(Field::new("item", elements.data_type().clone(), true)),
                    OffsetBuffer::new(vec![0i32, 3].into()),
                    elements,
                    None,
                ))
            };

            let mut from_hidden = vec![42u32; 1];
            create_murmur3_hashes(&[as_list(with_hidden)], &mut from_hidden).unwrap();
            let mut from_null = vec![42u32; 1];
            create_murmur3_hashes(&[as_list(without_hidden)], &mut from_null).unwrap();

            assert_eq!(
                from_hidden, from_null,
                "a null struct element must hash the same regardless of its child buffer"
            );
        }

        /// One `struct<a: Int32, b: Utf8>` element: `None` is a null struct, and the fields are
        /// independently nullable.
        type StructElem = Option<(Option<i32>, Option<&'static str>)>;
        /// One `array<struct<..>>` row: `None` is a null list.
        type ListRow = Option<Vec<StructElem>>;

        /// Builds `array<struct<a: Int32, b: Utf8>>` from `rows`.
        fn list_of_struct(rows: Vec<ListRow>) -> ArrayRef {
            use arrow::array::{Int32Builder, ListBuilder, StringBuilder, StructBuilder};
            use arrow::datatypes::{DataType, Field, Fields};

            let fields: Fields = vec![
                Arc::new(Field::new("a", DataType::Int32, true)),
                Arc::new(Field::new("b", DataType::Utf8, true)),
            ]
            .into();
            let struct_builder = StructBuilder::new(
                fields.clone(),
                vec![
                    Box::new(Int32Builder::new()),
                    Box::new(StringBuilder::new()),
                ],
            );
            let mut lb = ListBuilder::new(struct_builder);
            for row in rows {
                match row {
                    None => lb.append(false),
                    Some(elems) => {
                        for elem in elems {
                            let sb = lb.values();
                            match elem {
                                None => {
                                    sb.field_builder::<Int32Builder>(0)
                                        .unwrap()
                                        .append_null();
                                    sb.field_builder::<StringBuilder>(1)
                                        .unwrap()
                                        .append_null();
                                    sb.append(false);
                                }
                                Some((a, b)) => {
                                    match a {
                                        Some(v) => sb
                                            .field_builder::<Int32Builder>(0)
                                            .unwrap()
                                            .append_value(v),
                                        None => sb
                                            .field_builder::<Int32Builder>(0)
                                            .unwrap()
                                            .append_null(),
                                    }
                                    match b {
                                        Some(v) => sb
                                            .field_builder::<StringBuilder>(1)
                                            .unwrap()
                                            .append_value(v),
                                        None => sb
                                            .field_builder::<StringBuilder>(1)
                                            .unwrap()
                                            .append_null(),
                                    }
                                    sb.append(true);
                                }
                            }
                        }
                        lb.append(true);
                    }
                }
            }
            Arc::new(lb.finish())
        }

        fn hash_of(array: ArrayRef, num_rows: usize) -> Vec<u32> {
            let mut hashes = vec![42u32; num_rows];
            create_murmur3_hashes(&[array], &mut hashes).unwrap();
            hashes
        }

        /// `array<struct<..>>` is the shape that goes through `hash_list_array!`, the per-element path.
        /// These values were produced by that implementation and pin it: the hash decides which
        /// partition a row lands in, so any rewrite of that path has to reproduce them exactly.
        #[test]
        fn test_list_of_struct_hashes_are_stable() {
            let rows = vec![
                Some(vec![Some((Some(1), Some("x"))), Some((Some(2), Some("y")))]),
                Some(vec![Some((Some(3), Some("z")))]),
                // empty list: contributes nothing, so the seed survives
                Some(vec![]),
                // null list
                None,
                // null struct element, and elements with null fields
                Some(vec![None, Some((None, Some("w"))), Some((Some(4), None))]),
                // repeated element values, to catch an implementation that dedupes or reorders
                Some(vec![Some((Some(5), Some("s"))), Some((Some(5), Some("s")))]),
            ];
            let n = rows.len();
            assert_eq!(hash_of(list_of_struct(rows), n), EXPECTED_LIST_OF_STRUCT);
        }

        /// Element order must matter: Spark chains the element hashes in sequence.
        #[test]
        fn test_list_of_struct_is_order_sensitive() {
            let forward = list_of_struct(vec![Some(vec![
                Some((Some(1), Some("a"))),
                Some((Some(2), Some("b"))),
            ])]);
            let reversed = list_of_struct(vec![Some(vec![
                Some((Some(2), Some("b"))),
                Some((Some(1), Some("a"))),
            ])]);
            assert_ne!(
                hash_of(forward, 1),
                hash_of(reversed, 1),
                "element order must change the hash"
            );
        }

        /// One list far longer than the rest. A skewed batch must agree with hashing each row on its
        /// own.
        #[test]
        fn test_list_of_struct_skewed_lengths() {
            let mut rows: Vec<ListRow> = vec![Some(vec![Some((Some(1), Some("a")))]); 8];
            // One row far longer than the rest.
            rows.push(Some(
                (0..64)
                    .map(|i| Some((Some(i), Some("long"))))
                    .collect::<Vec<_>>(),
            ));
            rows.push(Some(vec![]));

            let batched = hash_of(list_of_struct(rows.clone()), rows.len());

            // Hash each row as its own batch of one; the result must match position by position.
            let per_row: Vec<u32> = rows
                .into_iter()
                .map(|row| hash_of(list_of_struct(vec![row]), 1)[0])
                .collect();
            assert_eq!(batched, per_row, "skewed batch must match per-row hashing");
        }

        /// `LargeList` has 64-bit offsets. The two list widths must agree on the same data.
        #[test]
        fn test_large_list_of_struct_matches_list() {
            use arrow::array::{
                Int32Builder, LargeListBuilder, StringBuilder, StructBuilder,
            };
            use arrow::datatypes::{DataType, Field, Fields};

            let fields: Fields = vec![
                Arc::new(Field::new("a", DataType::Int32, true)),
                Arc::new(Field::new("b", DataType::Utf8, true)),
            ]
            .into();
            let sb = StructBuilder::new(
                fields.clone(),
                vec![
                    Box::new(Int32Builder::new()),
                    Box::new(StringBuilder::new()),
                ],
            );
            let mut lb = LargeListBuilder::new(sb);
            for row in [vec![1, 2], vec![3], vec![]] {
                for v in row {
                    let s = lb.values();
                    s.field_builder::<Int32Builder>(0).unwrap().append_value(v);
                    s.field_builder::<StringBuilder>(1)
                        .unwrap()
                        .append_value(format!("s{v}"));
                    s.append(true);
                }
                lb.append(true);
            }
            let large: ArrayRef = Arc::new(lb.finish());

            let small = list_of_struct(vec![
                Some(vec![
                    Some((Some(1), Some("s1"))),
                    Some((Some(2), Some("s2"))),
                ]),
                Some(vec![Some((Some(3), Some("s3")))]),
                Some(vec![]),
            ]);

            assert_eq!(
                hash_of(large, 3),
                hash_of(small, 3),
                "LargeList and List must hash identically"
            );
        }

        /// `array<array<int>>`: the element is a list rather than a struct, so it takes the same
        /// non-primitive element path one level deeper.
        #[test]
        fn test_list_of_list_hashes_match_per_row() {
            use arrow::array::{Int32Builder, ListBuilder};

            let build = |rows: &[Vec<Vec<i32>>]| -> ArrayRef {
                let mut lb = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
                for row in rows {
                    for inner in row {
                        for v in inner {
                            lb.values().values().append_value(*v);
                        }
                        lb.values().append(true);
                    }
                    lb.append(true);
                }
                Arc::new(lb.finish())
            };

            let rows = vec![
                vec![vec![1, 2], vec![3]],
                vec![vec![4]],
                vec![],
                vec![vec![5, 6, 7], vec![], vec![8]],
            ];
            let batched = hash_of(build(&rows), rows.len());
            let per_row: Vec<u32> = rows
                .iter()
                .map(|row| hash_of(build(std::slice::from_ref(row)), 1)[0])
                .collect();
            assert_eq!(
                batched, per_row,
                "array<array<int>> must match per-row hashing"
            );
        }

        /// Rows that run out of elements at different positions, including in the middle of the
        /// batch, must agree with hashing each row as its own batch.
        #[test]
        fn test_list_of_struct_rows_exhaust_on_different_passes() {
            let rows: Vec<ListRow> = vec![
                Some((0..5).map(|i| Some((Some(i), Some("a")))).collect()),
                Some(vec![Some((Some(9), Some("b")))]),
                Some((0..3).map(|i| Some((Some(i), Some("c")))).collect()),
                Some(vec![]),
                Some((0..7).map(|i| Some((Some(i), Some("d")))).collect()),
                None,
                Some(vec![Some((Some(1), Some("e"))), Some((Some(2), Some("f")))]),
            ];
            let batched = hash_of(list_of_struct(rows.clone()), rows.len());
            let per_row: Vec<u32> = rows
                .into_iter()
                .map(|row| hash_of(list_of_struct(vec![row]), 1)[0])
                .collect();
            assert_eq!(
                batched, per_row,
                "rows exhausting on different passes must agree"
            );
        }

        /// A sliced list has a non-zero first offset, so an off-by-one in the offset handling would
        /// only show up on a slice.
        #[test]
        fn test_sliced_list_of_struct_matches_unsliced() {
            let rows: Vec<ListRow> = vec![
                Some(vec![Some((Some(1), Some("x")))]),
                Some(vec![Some((Some(2), Some("y"))), Some((Some(3), Some("z")))]),
                Some((0..4).map(|i| Some((Some(i), Some("w")))).collect()),
                Some(vec![Some((Some(8), Some("v")))]),
            ];
            let full = list_of_struct(rows.clone());

            // Hash rows 1..3 through a slice, and the same rows built on their own.
            let sliced = full.slice(1, 2);
            let mut from_slice = vec![42u32; 2];
            create_murmur3_hashes(&[sliced], &mut from_slice).unwrap();

            let standalone = list_of_struct(rows[1..3].to_vec());
            let mut from_standalone = vec![42u32; 2];
            create_murmur3_hashes(&[standalone], &mut from_standalone).unwrap();

            assert_eq!(
                from_slice, from_standalone,
                "a sliced list must hash like the same rows built unsliced"
            );
        }

        /// A null list can still cover a non-empty range of elements. Those elements must not be
        /// hashed.
        #[test]
        fn test_null_list_with_populated_range_is_skipped() {
            use arrow::array::{Int32Builder, ListArray, StringBuilder, StructBuilder};
            use arrow::buffer::{NullBuffer, OffsetBuffer};
            use arrow::datatypes::{DataType, Field, Fields};

            let fields: Fields = vec![
                Arc::new(Field::new("a", DataType::Int32, true)),
                Arc::new(Field::new("b", DataType::Utf8, true)),
            ]
            .into();
            let mut sb = StructBuilder::new(
                fields.clone(),
                vec![
                    Box::new(Int32Builder::new()),
                    Box::new(StringBuilder::new()),
                ],
            );
            for v in 0..3 {
                sb.field_builder::<Int32Builder>(0).unwrap().append_value(v);
                sb.field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_value("hidden");
                sb.append(true);
            }
            let elements: ArrayRef = Arc::new(sb.finish());

            // Row 0 covers elements 0..1, row 1 is null but still covers 1..3.
            let list = ListArray::new(
                Arc::new(Field::new("item", elements.data_type().clone(), true)),
                OffsetBuffer::new(vec![0i32, 1, 3].into()),
                elements,
                Some(NullBuffer::from(vec![true, false])),
            );
            let mut hashes = vec![42u32; 2];
            create_murmur3_hashes(&[Arc::new(list) as ArrayRef], &mut hashes).unwrap();
            assert_eq!(hashes[1], 42, "a null list must leave the seed untouched");

            // And it must not disturb the visible row either.
            let only_visible =
                list_of_struct(vec![Some(vec![Some((Some(0), Some("hidden")))])]);
            let mut expected = vec![42u32; 1];
            create_murmur3_hashes(&[only_visible], &mut expected).unwrap();
            assert_eq!(hashes[0], expected[0]);
        }

        /// Lengths that step down, rather than being all equal or one long outlier.
        #[test]
        fn test_list_of_struct_descending_lengths() {
            let rows: Vec<ListRow> = (1..=12)
                .rev()
                .map(|n| Some((0..n).map(|i| Some((Some(i), Some("s")))).collect()))
                .collect();
            let batched = hash_of(list_of_struct(rows.clone()), rows.len());
            let per_row: Vec<u32> = rows
                .into_iter()
                .map(|row| hash_of(list_of_struct(vec![row]), 1)[0])
                .collect();
            assert_eq!(batched, per_row);
        }

        /// A lone-row list must not stop the caller hashing the remaining columns.
        #[test]
        fn test_single_row_list_then_another_column() {
            let l = list_of_struct(vec![Some(vec![
                Some((Some(1), Some("a"))),
                Some((Some(2), Some("b"))),
            ])]);
            let other: ArrayRef = Arc::new(Int32Array::from(vec![7]));

            // both columns together
            let mut both = vec![42u32; 1];
            create_murmur3_hashes(&[Arc::clone(&l), Arc::clone(&other)], &mut both)
                .unwrap();

            // chaining them by hand must agree
            let mut step = vec![42u32; 1];
            create_murmur3_hashes(&[l], &mut step).unwrap();
            create_murmur3_hashes(&[other], &mut step).unwrap();

            assert_eq!(both, step, "the second column must still be hashed");
        }

        #[test]
        fn test_i8() {
            test_murmur3_hash::<i8, Int8Array>(
                vec![Some(1), Some(0), Some(-1), Some(i8::MAX), Some(i8::MIN)],
                vec![0xdea578e3, 0x379fae8f, 0xa0590e3d, 0x43b4d8ed, 0x422a1365],
            );
        }

        #[test]
        fn test_i32() {
            test_murmur3_hash::<i32, Int32Array>(
                vec![Some(1), Some(0), Some(-1), Some(i32::MAX), Some(i32::MIN)],
                vec![0xdea578e3, 0x379fae8f, 0xa0590e3d, 0x07fb67e7, 0x2b1f0fc6],
            );
        }

        #[test]
        fn test_i64() {
            test_murmur3_hash::<i64, Int64Array>(
                vec![Some(1), Some(0), Some(-1), Some(i64::MAX), Some(i64::MIN)],
                vec![0x99f0149d, 0x9c67b85d, 0xc8008529, 0xa05b5d7b, 0xcd1e64fb],
            );
        }

        /// Spark hashes `YearMonthIntervalType` as an int and `DayTimeIntervalType` as a long, which
        /// Comet maps to `Interval(YearMonth)` and `Duration(Microsecond)`.
        #[test]
        fn test_ansi_intervals_hash_like_int_and_long() {
            use arrow::array::{DurationMicrosecondArray, IntervalYearMonthArray};

            test_murmur3_hash::<i32, IntervalYearMonthArray>(
                vec![Some(1), Some(0), Some(-1), Some(i32::MAX), Some(i32::MIN)],
                vec![0xdea578e3, 0x379fae8f, 0xa0590e3d, 0x07fb67e7, 0x2b1f0fc6],
            );
            test_murmur3_hash::<i64, DurationMicrosecondArray>(
                vec![Some(1), Some(0), Some(-1), Some(i64::MAX), Some(i64::MIN)],
                vec![0x99f0149d, 0x9c67b85d, 0xc8008529, 0xa05b5d7b, 0xcd1e64fb],
            );
        }

        /// A `NullType` value is always null, so it leaves the hash unchanged, as in Spark, both as a
        /// column of its own and as a struct field, including a field of a null struct.
        #[test]
        fn test_null_type_leaves_hash_unchanged() {
            use arrow::array::{NullArray, StructArray};
            use arrow::buffer::NullBuffer;
            use arrow::datatypes::{DataType, Field};

            let ints: ArrayRef =
                Arc::new(Int32Array::from(vec![Some(1), None, Some(-1)]));
            let nulls: ArrayRef = Arc::new(NullArray::new(3));

            let mut expected = vec![42u32; 3];
            create_murmur3_hashes(&[Arc::clone(&ints)], &mut expected).unwrap();
            let mut hashes = vec![42u32; 3];
            create_murmur3_hashes(&[Arc::clone(&ints), Arc::clone(&nulls)], &mut hashes)
                .unwrap();
            assert_eq!(hashes, expected);

            let struct_nulls = Some(NullBuffer::from(vec![true, true, false]));
            let x = Arc::new(Field::new("x", DataType::Int32, true));
            let n = Arc::new(Field::new("n", DataType::Null, true));
            let with_null_field: ArrayRef = Arc::new(StructArray::new(
                vec![Arc::clone(&x), n].into(),
                vec![Arc::clone(&ints), nulls],
                struct_nulls.clone(),
            ));
            let without_null_field: ArrayRef =
                Arc::new(StructArray::new(vec![x].into(), vec![ints], struct_nulls));

            let mut expected = vec![42u32; 3];
            create_murmur3_hashes(&[without_null_field], &mut expected).unwrap();
            let mut hashes = vec![42u32; 3];
            create_murmur3_hashes(&[with_null_field], &mut hashes).unwrap();
            assert_eq!(hashes, expected);
            assert_eq!(hashes[2], 42, "a null struct must leave the seed untouched");
        }

        /// Lists whose elements are `NullType` or an ANSI interval type go through the recursive
        /// list path into the `Null`, `Interval(YearMonth)` and `Duration(Microsecond)` arms. A list
        /// of nulls leaves the hash unchanged, and a list of intervals hashes like the same list of
        /// ints or longs.
        #[test]
        fn test_lists_of_null_type_and_intervals() {
            use arrow::array::{
                DurationMicrosecondArray, IntervalYearMonthArray, ListArray, NullArray,
            };
            use arrow::buffer::{NullBuffer, OffsetBuffer};
            use arrow::datatypes::Field;

            // Rows [v0, v1], [], null and [v2].
            let as_list = |values: ArrayRef| -> ArrayRef {
                Arc::new(ListArray::new(
                    Arc::new(Field::new("item", values.data_type().clone(), true)),
                    OffsetBuffer::from_lengths([2, 0, 0, 1]),
                    values,
                    Some(NullBuffer::from(vec![true, true, false, true])),
                ))
            };
            let hash = |list: ArrayRef| {
                let mut hashes = vec![42u32; 4];
                create_murmur3_hashes(&[list], &mut hashes).unwrap();
                hashes
            };

            assert_eq!(hash(as_list(Arc::new(NullArray::new(3)))), vec![42u32; 4]);

            let months = vec![Some(1), None, Some(i32::MIN)];
            assert_eq!(
                hash(as_list(Arc::new(IntervalYearMonthArray::from(
                    months.clone()
                )))),
                hash(as_list(Arc::new(Int32Array::from(months))))
            );
            let micros = vec![Some(-1), Some(i64::MAX), None];
            assert_eq!(
                hash(as_list(Arc::new(DurationMicrosecondArray::from(
                    micros.clone()
                )))),
                hash(as_list(Arc::new(Int64Array::from(micros))))
            );
        }

        #[test]
        fn test_f32() {
            test_murmur3_hash::<f32, Float32Array>(
                vec![
                    Some(1.0),
                    Some(0.0),
                    Some(-0.0),
                    Some(-1.0),
                    Some(99999999999.99999999999),
                    Some(-99999999999.99999999999),
                    // Every NaN hashes as the canonical NaN, whatever its sign or payload.
                    Some(f32::NAN),
                    Some(f32::from_bits(0xffc0_0000)),
                    Some(f32::from_bits(0x7f80_0001)),
                ],
                vec![
                    0xe434cc39, 0x379fae8f, 0x379fae8f, 0xdc0da8eb, 0xcbdc340f,
                    0xc0361c86, 0xeb2eb18a, 0xeb2eb18a, 0xeb2eb18a,
                ],
            );
        }

        #[test]
        fn test_f64() {
            test_murmur3_hash::<f64, Float64Array>(
                vec![
                    Some(1.0),
                    Some(0.0),
                    Some(-0.0),
                    Some(-1.0),
                    Some(99999999999.99999999999),
                    Some(-99999999999.99999999999),
                    // Every NaN hashes as the canonical NaN, whatever its sign or payload.
                    Some(f64::NAN),
                    Some(f64::from_bits(0xfff8_0000_0000_0000)),
                    Some(f64::from_bits(0x7ff0_0000_0000_0001)),
                ],
                vec![
                    0xe4876492, 0x9c67b85d, 0x9c67b85d, 0x13d81357, 0xb87e1595,
                    0xa0eef9f9, 0xb3a005cf, 0xb3a005cf, 0xb3a005cf,
                ],
            );
        }

        /// Spark hashes a float through `doubleToLongBits` or `floatToIntBits`, which canonicalize
        /// NaN, so a NaN with the sign bit set or with a payload hashes like the canonical NaN inside
        /// lists, structs, maps and dictionaries too. Checks both hashes, which expand the same macros.
        #[test]
        fn test_non_canonical_nan_hashes_like_canonical_nan() {
            use arrow::array::{
                DictionaryArray, FixedSizeListArray, Int32Array, LargeListArray,
                ListArray, MapArray, StructArray,
            };
            use arrow::buffer::OffsetBuffer;
            use arrow::datatypes::{DataType, Field, Fields};

            let shapes = |values: ArrayRef| -> Vec<ArrayRef> {
                let field =
                    Arc::new(Field::new("item", values.data_type().clone(), true));
                // One entry per row, with the float as both the key and the value.
                let entry_fields = Fields::from(vec![
                    Field::new("key", values.data_type().clone(), false),
                    Field::new("value", values.data_type().clone(), true),
                ]);
                let entries = StructArray::new(
                    entry_fields.clone(),
                    vec![Arc::clone(&values), Arc::clone(&values)],
                    None,
                );
                vec![
                    Arc::clone(&values),
                    Arc::new(ListArray::new(
                        Arc::clone(&field),
                        OffsetBuffer::from_lengths([1; 3]),
                        Arc::clone(&values),
                        None,
                    )),
                    Arc::new(LargeListArray::new(
                        Arc::clone(&field),
                        OffsetBuffer::from_lengths([1; 3]),
                        Arc::clone(&values),
                        None,
                    )),
                    Arc::new(FixedSizeListArray::new(
                        Arc::clone(&field),
                        1,
                        Arc::clone(&values),
                        None,
                    )),
                    Arc::new(StructArray::new(
                        vec![field].into(),
                        vec![Arc::clone(&values)],
                        None,
                    )),
                    Arc::new(DictionaryArray::new(
                        Int32Array::from(vec![0, 1, 2]),
                        values,
                    )),
                    Arc::new(MapArray::new(
                        Arc::new(Field::new(
                            "entries",
                            DataType::Struct(entry_fields),
                            false,
                        )),
                        OffsetBuffer::from_lengths([1; 3]),
                        entries,
                        None,
                        false,
                    )),
                ]
            };
            let canonical64: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN; 3]));
            let other64: ArrayRef = Arc::new(Float64Array::from(vec![
                f64::from_bits(0xfff8_0000_0000_0000),
                f64::from_bits(0x7ff0_0000_0000_0001),
                f64::from_bits(0xfff0_0000_0000_0001),
            ]));
            let canonical32: ArrayRef = Arc::new(Float32Array::from(vec![f32::NAN; 3]));
            let other32: ArrayRef = Arc::new(Float32Array::from(vec![
                f32::from_bits(0xffc0_0000),
                f32::from_bits(0x7f80_0001),
                f32::from_bits(0xff80_0001),
            ]));
            for (canonical, other) in [(canonical64, other64), (canonical32, other32)] {
                for (canonical, other) in shapes(canonical).into_iter().zip(shapes(other))
                {
                    let (mut expected, mut actual) = (vec![42u32; 3], vec![42u32; 3]);
                    create_murmur3_hashes(&[Arc::clone(&canonical)], &mut expected)
                        .unwrap();
                    create_murmur3_hashes(&[Arc::clone(&other)], &mut actual).unwrap();
                    assert_eq!(actual, expected, "murmur3 of {}", other.data_type());
                    let (mut expected, mut actual) = (vec![42u64; 3], vec![42u64; 3]);
                    create_xxhash64_hashes(&[canonical], &mut expected).unwrap();
                    create_xxhash64_hashes(&[Arc::clone(&other)], &mut actual).unwrap();
                    assert_eq!(actual, expected, "xxhash64 of {}", other.data_type());
                }
            }
        }

        #[test]
        fn test_str() {
            let input = [
                "hello", "bar", "", "😁", "天地", "a", "ab", "abc", "abcd", "abcde",
            ]
            .iter()
            .map(|s| Some(s.to_string()))
            .collect::<Vec<Option<String>>>();
            let expected: Vec<u32> = vec![
                3286402344, 2486176763, 142593372, 885025535, 2395000894, 1485273170,
                0xfa37157b, 1322437556, 0xe860e5cc, 814637928,
            ];

            test_murmur3_hash::<String, StringArray>(input.clone(), expected);
        }

        // `create_hashes_internal!` specialises maps of (Utf8, Int32), (Int32, Utf8), (Utf8, Utf8)
        // and (Int32, Int32), each with a loop for input without nulls and a null-aware loop.
        // `sql-tests/expressions/hash/hash.sql` checks both loops of every pair against Spark. The
        // tests below cover the shapes SQL cannot build by comparing with the same rows with that map row removed

        /// (key is Utf8, value is Utf8) for each specialised pair.
        const MAP_PAIRS: [(bool, bool); 4] =
            [(true, false), (false, true), (true, true), (false, false)];

        /// Entries per row of every test map: row 2 owns entry 2.
        const MAP_LENGTHS: [usize; 4] = [2, 0, 1, 3];
        /// Map validity per row. With every map valid the hash takes the loop for input without
        /// nulls; row 2 being null sends it to the null-aware loop.
        const MAP_ALL_VALID: [bool; 4] = [true; 4];
        const MAP_NULL_ROW: [bool; 4] = [true, true, false, true];

        /// The seed every row starts from. Spark leaves it unchanged for a null map.
        const SEED: u32 = 42;

        fn map_pair_label(key_utf8: bool, value_utf8: bool) -> String {
            let name = |utf8: bool| if utf8 { "Utf8" } else { "Int32" };
            format!("({}, {})", name(key_utf8), name(value_utf8))
        }

        /// The first `len` entries of a pool, as a Utf8 or Int32 column.
        fn map_entry_column(
            utf8: bool,
            strs: &[&str],
            ints: &[i32],
            len: usize,
        ) -> ArrayRef {
            if utf8 {
                Arc::new(StringArray::from(strs[..len].to_vec()))
            } else {
                Arc::new(Int32Array::from(ints[..len].to_vec()))
            }
        }

        /// A map with `MAP_LENGTHS` entries per row and the given row validity. Every entry is
        /// distinct, so hashing the wrong entry changes the hash.
        fn map_of(key_utf8: bool, value_utf8: bool, valid: &[bool]) -> ArrayRef {
            use arrow::array::{MapArray, StructArray};
            use arrow::buffer::{NullBuffer, OffsetBuffer};
            use arrow::datatypes::{DataType, Field, Fields};

            const STR_KEYS: [&str; 6] = ["a", "b", "", "日本", "x", "yz"];
            const INT_KEYS: [i32; 6] = [1, -1, 0, i32::MAX, i32::MIN, 7];
            const STR_VALUES: [&str; 6] = ["v1", "", "é", "w", "long value", "z"];
            const INT_VALUES: [i32; 6] = [10, -20, 0, i32::MIN, i32::MAX, 3];

            let len = MAP_LENGTHS.iter().sum();
            let keys = map_entry_column(key_utf8, &STR_KEYS, &INT_KEYS, len);
            let values = map_entry_column(value_utf8, &STR_VALUES, &INT_VALUES, len);
            let entry_fields = Fields::from(vec![
                Field::new("key", keys.data_type().clone(), false),
                Field::new("value", values.data_type().clone(), true),
            ]);
            let entries =
                StructArray::new(entry_fields.clone(), vec![keys, values], None);
            Arc::new(MapArray::new(
                Arc::new(Field::new("entries", DataType::Struct(entry_fields), false)),
                OffsetBuffer::from_lengths(MAP_LENGTHS),
                entries,
                valid
                    .contains(&false)
                    .then(|| NullBuffer::from(valid.to_vec())),
                false,
            ))
        }

        /// murmur3 and xxhash64 of every row, both starting from `SEED`.
        fn map_hashes(array: &ArrayRef) -> (Vec<u32>, Vec<u64>) {
            let mut murmur3 = vec![SEED; array.len()];
            create_murmur3_hashes(&[Arc::clone(array)], &mut murmur3).unwrap();
            let mut xxhash64 = vec![SEED as u64; array.len()];
            create_xxhash64_hashes(&[Arc::clone(array)], &mut xxhash64).unwrap();
            (murmur3, xxhash64)
        }

        /// Whether the hash takes the null-aware loop. Like the loop selection itself, this counts
        /// the nulls of the whole key and value arrays, not only the entries a slice covers.
        fn map_takes_null_aware_loop(array: &ArrayRef) -> bool {
            use arrow::array::{Array, MapArray};

            let map = array.as_any().downcast_ref::<MapArray>().unwrap();
            map.null_count() + map.keys().null_count() + map.values().null_count() > 0
        }

        /// A null map can still own entries, which Spark never produces but Arrow allows. Those
        /// entries must not be hashed: the null row keeps the seed, and the other rows hash as they
        /// do once the null row, and with it the entry it owns, is filtered out.
        #[test]
        fn test_specialised_map_null_row_skips_owned_entries() {
            use arrow::array::BooleanArray;
            use arrow::compute::filter;

            let null_row = MAP_NULL_ROW.iter().position(|v| !v).unwrap();
            assert!(MAP_LENGTHS[null_row] > 0, "the null map must own an entry");
            let keep = BooleanArray::from(MAP_NULL_ROW.to_vec());

            for (key_utf8, value_utf8) in MAP_PAIRS {
                let label = map_pair_label(key_utf8, value_utf8);
                let array = map_of(key_utf8, value_utf8, &MAP_NULL_ROW);
                assert!(map_takes_null_aware_loop(&array), "{label}");

                let (mut murmur3, mut xxhash64) = map_hashes(&array);
                assert_eq!(
                    murmur3.remove(null_row),
                    SEED,
                    "murmur3 of the null map, {label}"
                );
                assert_eq!(
                    xxhash64.remove(null_row),
                    SEED as u64,
                    "xxhash64 of the null map, {label}"
                );

                let without_null_row = filter(&array, &keep).unwrap();
                let (expected_murmur3, expected_xxhash64) = map_hashes(&without_null_row);
                assert_eq!(
                    murmur3, expected_murmur3,
                    "murmur3 of the other rows, {label}"
                );
                assert_eq!(
                    xxhash64, expected_xxhash64,
                    "xxhash64 of the other rows, {label}"
                );
            }
        }

        /// A sliced map keeps its unsliced keys and values, so its offsets do not start at 0. Each
        /// row must hash as it does in the unsliced array, in both loops.
        #[test]
        fn test_specialised_map_slice_matches_unsliced() {
            use arrow::array::{Array, MapArray};

            for (key_utf8, value_utf8) in MAP_PAIRS {
                for valid in [MAP_ALL_VALID, MAP_NULL_ROW] {
                    let has_nulls = valid.contains(&false);
                    let label = format!(
                        "{} {}",
                        map_pair_label(key_utf8, value_utf8),
                        if has_nulls {
                            "with a null map"
                        } else {
                            "without nulls"
                        }
                    );
                    let array = map_of(key_utf8, value_utf8, &valid);
                    let sliced = array.slice(1, array.len() - 1);

                    let offsets = sliced
                        .as_any()
                        .downcast_ref::<MapArray>()
                        .unwrap()
                        .offsets();
                    assert_ne!(
                        offsets[0], 0,
                        "{label}: the slice must not start at entry 0"
                    );
                    assert_eq!(map_takes_null_aware_loop(&sliced), has_nulls, "{label}");

                    let (murmur3, xxhash64) = map_hashes(&array);
                    let (sliced_murmur3, sliced_xxhash64) = map_hashes(&sliced);
                    assert_eq!(sliced_murmur3[..], murmur3[1..], "murmur3 of {label}");
                    assert_eq!(sliced_xxhash64[..], xxhash64[1..], "xxhash64 of {label}");
                }
            }
        }
    }
}
