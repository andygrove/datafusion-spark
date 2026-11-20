// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use crate::function::functions_nested_utils::list_element_field;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Capacities, GenericListArray, Int32Array, MutableArrayData,
    OffsetSizeTrait, make_array,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, FieldRef, Int32Type};
use datafusion_common::utils::take_function_args;
use datafusion_common::{
    DataFusionError, Result, ScalarValue, exec_err, internal_err, plan_datafusion_err,
    plan_err,
};
use datafusion_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::type_coercion::{
    spark_sql_type, struct_field_types_match, tightest_common_type,
};

/// `ByteArrayUtils.MAX_ROUNDED_ARRAY_LENGTH` (`Integer.MAX_VALUE - 15`) in Spark.
const MAX_ROUNDED_ARRAY_LENGTH: i64 = 2_147_483_632;

/// Spark-compatible `array_insert` expression.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#array_insert>
///
/// Use [`Self::with_legacy_negative_index`] for Spark's
/// `spark.sql.legacy.negativeIndexInArrayInsert`.
#[user_doc(
    doc_section(label = "Array Functions"),
    description = "Returns `array` with `item` inserted at index `pos`. DataFusion has no built-in `array_insert`. \
        Indexes are 1-based, and a negative index counts from the end, so -1 inserts after the last element. \
        An index past the end pads with NULLs before the item. A negative index before the start places the item first, \
        then NULL padding, then the original elements. Returns NULL if `array` or `pos` is NULL. \
        An index of 0 is an error (`INVALID_INDEX_OF_ZERO`), as is a result longer than Spark's maximum array length \
        (`COLLECTION_SIZE_LIMIT_EXCEEDED`). With Spark's `spark.sql.legacy.negativeIndexInArrayInsert`, \
        -1 inserts before the last element instead, and a negative index before the start pads one more NULL, \
        so -(n+1) on an array of n elements gives the item, one NULL, then the elements.",
    syntax_example = "array_insert(array, pos, item)",
    sql_example = r#"```sql
> SELECT array_insert(make_array(1, 2, 3), 2, 10);
[1, 10, 2, 3]
> SELECT array_insert(make_array(5, 4, 3, 2), -1, 1);
[5, 4, 3, 2, 1]
> SELECT array_insert(make_array(1, 2, 3), 5, 100);
[1, 2, 3, NULL, 100]
> SELECT array_insert(make_array(1), -3, 100);
[100, NULL, 1]
> SELECT array_insert(make_array(1, 2), NULL::INT, 5);
NULL
```"#,
    argument(
        name = "array",
        description = "Array expression to insert into. NULL returns NULL."
    ),
    argument(
        name = "pos",
        description = "Integer index, 1-based, of any integer type (cast to INT). A negative value counts from the end. 0 is an error. NULL returns NULL."
    ),
    argument(
        name = "item",
        description = "Value to insert. Must share a type family with the array's elements. A NULL item is inserted as a NULL element."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkArrayInsert {
    signature: Signature,
    legacy_negative_index: bool,
}

impl Default for SparkArrayInsert {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkArrayInsert {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            legacy_negative_index: false,
        }
    }

    /// Enables Spark's legacy negative index behavior
    /// (`spark.sql.legacy.negativeIndexInArrayInsert`): -1 inserts before the
    /// last element, and a negative index before the start pads one more NULL.
    ///
    /// For example `SparkArrayInsert::new().with_legacy_negative_index(true)`.
    /// Once `SparkOptions` lands (#92), a DataFusion session would read this
    /// from `spark.sql.legacy.negativeIndexInArrayInsert` instead.
    pub fn with_legacy_negative_index(mut self, legacy_negative_index: bool) -> Self {
        self.legacy_negative_index = legacy_negative_index;
        self
    }
}

/// Spark's error when the first argument is not an array, or the item does
/// not share a type with the array's elements.
fn diff_types_err(array: &DataType, item: &DataType) -> DataFusionError {
    plan_datafusion_err!(
        "[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES] Input to `array_insert` should have been \"ARRAY\" followed by a value with same element type, but it's [\"{}\", \"{}\"]. SQLSTATE: 42K09",
        spark_sql_type(array),
        spark_sql_type(item)
    )
}

impl ScalarUDFImpl for SparkArrayInsert {
    fn name(&self) -> &str {
        "array_insert"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        let [array, pos, _item] = take_function_args(self.name(), args.arg_fields)?;
        let Some(elem) = list_element_field(array.data_type()) else {
            return internal_err!(
                "array_insert expects a list, got {}",
                array.data_type()
            );
        };
        // Padding can introduce NULLs even when neither input has any, so the
        // element field is always nullable.
        let elem = Arc::new(Field::new(elem.name(), elem.data_type().clone(), true));
        let data_type = match array.data_type() {
            DataType::LargeList(_) => DataType::LargeList(elem),
            _ => DataType::List(elem),
        };
        Ok(Arc::new(Field::new(
            self.name(),
            data_type,
            array.is_nullable() || pos.is_nullable(),
        )))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [array, pos, item] = take_function_args(self.name(), arg_types)?;

        let (elem, wrap): (&FieldRef, fn(FieldRef) -> DataType) = match array {
            DataType::List(f) | DataType::FixedSizeList(f, _) => (f, DataType::List),
            DataType::LargeList(f) => (f, DataType::LargeList),
            _ => return Err(diff_types_err(array, item)),
        };

        // Spark takes an INT index. DataFusion types bare integer literals as
        // BIGINT, so accept every integer type and cast to INT; a value outside
        // the INT range then fails the cast instead of being silently truncated.
        if !pos.is_integer() {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The second parameter of `array_insert` requires the \"INT\" type, however it has the type \"{}\". SQLSTATE: 42K09",
                spark_sql_type(pos)
            );
        }

        // Spark's `ArrayInsert` casts both to the non-ANSI
        // `TypeCoercion.findTightestCommonType` of the element and the item. That
        // unifies Utf8 and Utf8View (parquet's default for strings), timestamps
        // of different units, and nested types that differ only in nullability,
        // and pairs struct fields by position.
        let elem_type = elem.data_type();
        let common = tightest_common_type(elem_type, item, false)
            .filter(|_| struct_field_types_match(elem_type, item));
        let Some(common) = common else {
            return Err(diff_types_err(array, item));
        };

        let new_elem =
            Arc::new(Field::new(elem.name(), common.clone(), elem.is_nullable()));
        Ok(vec![wrap(new_elem), DataType::Int32, common])
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let all_scalar = args
            .args
            .iter()
            .all(|arg| matches!(arg, ColumnarValue::Scalar(_)));
        let arrays = ColumnarValue::values_to_arrays(&args.args)?;
        let [list, pos, item] = take_function_args(self.name(), arrays)?;
        let Some(pos) = pos.as_primitive_opt::<Int32Type>() else {
            return internal_err!(
                "array_insert expects an Int32 index, got {}",
                pos.data_type()
            );
        };
        let legacy = self.legacy_negative_index;
        let Some(elem_field) = list_element_field(args.return_field.data_type()) else {
            return internal_err!(
                "unexpected array_insert return type {}",
                args.return_field.data_type()
            );
        };
        let elem_field = Arc::clone(elem_field);
        let result: ArrayRef = match list.data_type() {
            DataType::List(_) => Arc::new(array_insert(
                list.as_list::<i32>(),
                &item,
                pos,
                legacy,
                elem_field,
            )?),
            DataType::LargeList(_) => Arc::new(array_insert(
                list.as_list::<i64>(),
                &item,
                pos,
                legacy,
                elem_field,
            )?),
            other => return internal_err!("array_insert expects a list, got {other}"),
        };
        if all_scalar {
            Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                &result, 0,
            )?))
        } else {
            Ok(ColumnarValue::Array(result))
        }
    }
}

/// Where the item goes in a row, and how long the row becomes.
struct Placement {
    /// Original elements copied before the item.
    before: usize,
    /// NULLs between the item and the original elements (negative index
    /// before the start of the array).
    leading_nulls: usize,
    /// NULLs between the original elements and the item (positive index past
    /// the end of the array).
    trailing_nulls: usize,
    /// Length of the resulting row.
    new_len: usize,
}

fn place(pos: i32, len: usize, legacy: bool) -> Result<Placement> {
    if pos == 0 {
        return exec_err!(
            "[INVALID_INDEX_OF_ZERO] The index 0 is invalid. An index shall be either < 0 or > 0 (the first element has index 1). SQLSTATE: 22003"
        );
    }
    let n = len as i64;
    let pos = i64::from(pos);
    // (original elements before the item, leading NULLs, trailing NULLs, new length)
    let (before, leading, trailing, new_len) = if pos > 0 {
        let idx = pos - 1;
        if idx <= n {
            (idx, 0, 0, n + 1)
        } else {
            (n, 0, idx - n, pos)
        }
    } else if -pos > n {
        let new_len = -pos + i64::from(legacy);
        (0, new_len - n - 1, 0, new_len)
    } else {
        (n + pos + i64::from(!legacy), 0, 0, n + 1)
    };
    if new_len > MAX_ROUNDED_ARRAY_LENGTH {
        return exec_err!(
            "[COLLECTION_SIZE_LIMIT_EXCEEDED.FUNCTION] Can't create array with {new_len} elements which exceeding the array size limit {MAX_ROUNDED_ARRAY_LENGTH}, unsuccessful try to create arrays in the function `array_insert`. SQLSTATE: 54000"
        );
    }
    Ok(Placement {
        before: before as usize,
        leading_nulls: leading as usize,
        trailing_nulls: trailing as usize,
        new_len: new_len as usize,
    })
}

fn array_insert<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
    item: &ArrayRef,
    pos: &Int32Array,
    legacy: bool,
    elem_field: FieldRef,
) -> Result<GenericListArray<O>> {
    // First pass: validate every row and compute the output offsets, so the
    // child data can be allocated once at its final size.
    let mut placements = Vec::with_capacity(list.len());
    let mut offsets = Vec::with_capacity(list.len() + 1);
    offsets.push(O::usize_as(0));
    let mut total = 0usize;
    for row in 0..list.len() {
        // A NULL array or NULL index gives a NULL row with no child values.
        let placement = if list.is_null(row) || pos.is_null(row) {
            None
        } else {
            let p = place(pos.value(row), list.value_length(row).as_usize(), legacy)?;
            total += p.new_len;
            Some(p)
        };
        placements.push(placement);
        // A `List` (i32 offsets) can overflow across rows even though each row is
        // within Spark's per-array limit.
        let Some(offset) = O::from_usize(total) else {
            return exec_err!(
                "array_insert result is too large for the list's offset type"
            );
        };
        offsets.push(offset);
    }

    let values = list.values().to_data();
    let item_data = item.to_data();
    // `MutableArrayData` panics on mixed types. SQL coerces the item to the
    // element type, but a caller that builds the physical expression directly,
    // such as Comet, may not.
    if values.data_type() != item_data.data_type() {
        return internal_err!(
            "array_insert expects the item type {} to match the element type {}",
            item_data.data_type(),
            values.data_type()
        );
    }
    let mut out = MutableArrayData::with_capacities(
        vec![&values, &item_data],
        true,
        Capacities::Array(total),
    );
    for (row, p) in placements.iter().enumerate() {
        let Some(p) = p else { continue };
        let start = list.value_offsets()[row].as_usize();
        let end = list.value_offsets()[row + 1].as_usize();
        out.try_extend(0, start, start + p.before)?;
        if p.trailing_nulls > 0 {
            // Positive index past the end: original elements, NULLs, item.
            out.try_extend(0, start + p.before, end)?;
            out.try_extend_nulls(p.trailing_nulls)?;
            out.try_extend(1, row, row + 1)?;
        } else {
            // Otherwise: item, then (for a negative index before the start)
            // NULL padding, then the remaining original elements.
            out.try_extend(1, row, row + 1)?;
            out.try_extend_nulls(p.leading_nulls)?;
            out.try_extend(0, start + p.before, end)?;
        }
    }

    GenericListArray::try_new(
        elem_field,
        OffsetBuffer::new(offsets.into()),
        make_array(out.freeze()),
        NullBuffer::union(list.nulls(), pos.nulls()),
    )
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::ListArray;
    use arrow::datatypes::{Int32Type, TimeUnit};

    fn run(
        rows: Vec<Option<Vec<Option<i32>>>>,
        pos: Vec<Option<i32>>,
        items: Vec<Option<i32>>,
        legacy: bool,
    ) -> Result<ListArray> {
        let list = ListArray::from_iter_primitive::<Int32Type, _, _>(rows);
        let field = Arc::new(Field::new_list_field(DataType::Int32, true));
        let item: ArrayRef = Arc::new(Int32Array::from(items));
        array_insert(&list, &item, &Int32Array::from(pos), legacy, field)
    }

    fn list(rows: Vec<Option<Vec<Option<i32>>>>) -> ListArray {
        ListArray::from_iter_primitive::<Int32Type, _, _>(rows)
    }

    // Most behavior is covered by array_insert.slt. These cover what SQL
    // cannot reach: legacy mode (a session config), `i32::MIN`, and sliced
    // input. Expected values were verified with pyspark==4.2.0.
    #[test]
    fn legacy_builder_through_udf() -> Result<()> {
        let list_type =
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true)));
        let invoke = |udf: SparkArrayInsert| {
            udf.invoke_with_args(ScalarFunctionArgs {
                args: vec![
                    ColumnarValue::Scalar(ScalarValue::List(Arc::new(list(vec![Some(
                        vec![Some(1), Some(2), Some(3)],
                    )])))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(-1))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(10))),
                ],
                arg_fields: vec![],
                number_rows: 1,
                return_field: Arc::new(Field::new("r", list_type.clone(), true)),
                config_options: Default::default(),
            })
        };
        let expect = |values: Vec<Option<i32>>| {
            ScalarValue::List(Arc::new(list(vec![Some(values)])))
        };
        // All-scalar input gives a scalar result.
        let ColumnarValue::Scalar(got) = invoke(SparkArrayInsert::new())? else {
            panic!("expected a scalar result");
        };
        assert_eq!(got, expect(vec![Some(1), Some(2), Some(3), Some(10)]));
        let ColumnarValue::Scalar(got) =
            invoke(SparkArrayInsert::new().with_legacy_negative_index(true))?
        else {
            panic!("expected a scalar result");
        };
        assert_eq!(got, expect(vec![Some(1), Some(2), Some(10), Some(3)]));
        Ok(())
    }

    #[test]
    fn legacy_negative_index() -> Result<()> {
        let got = run(
            vec![
                Some(vec![Some(1), Some(2), Some(3)]),
                Some(vec![Some(4), Some(5)]),
                None,
                // before the start: the item, four NULLs, then the elements
                Some(vec![Some(1)]),
                // -(len + 1) pads one NULL in legacy mode
                Some(vec![Some(1), Some(2), Some(3)]),
                // -len inserts first with no padding in legacy mode
                Some(vec![Some(1), Some(2), Some(3)]),
            ],
            vec![Some(-1), Some(-1), Some(-1), Some(-5), Some(-4), Some(-3)],
            vec![Some(10), Some(20), Some(30), Some(9), Some(10), Some(10)],
            true,
        )?;
        let expected = list(vec![
            Some(vec![Some(1), Some(2), Some(10), Some(3)]),
            Some(vec![Some(4), Some(20), Some(5)]),
            None,
            Some(vec![Some(9), None, None, None, None, Some(1)]),
            Some(vec![Some(10), None, Some(1), Some(2), Some(3)]),
            Some(vec![Some(10), Some(1), Some(2), Some(3)]),
        ]);
        assert_eq!(got.to_data(), expected.to_data());
        Ok(())
    }

    #[test]
    fn min_index_exceeds_size_limit() {
        // -i32::MIN overflows i32. Spark 4.2.0 reports 2147483648 (2147483649 in
        // legacy mode) since SPARK-56567; Spark 4.1.x and older overflow instead.
        for (legacy, size) in [(false, "2147483648"), (true, "2147483649")] {
            let err = run(
                vec![Some(vec![Some(1), Some(2), Some(3)])],
                vec![Some(i32::MIN)],
                vec![Some(10)],
                legacy,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("COLLECTION_SIZE_LIMIT_EXCEEDED"), "{err}");
            assert!(
                err.contains(&format!("create array with {size} elements")),
                "{err}"
            );
        }
    }

    #[test]
    fn sliced_list() -> Result<()> {
        // Row offsets of a sliced list do not start at zero.
        let full = list(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4), Some(5)]),
            None,
            Some(vec![Some(1)]),
            Some(vec![Some(7)]),
        ]);
        let sliced = full.slice(1, 3);
        let field = Arc::new(Field::new_list_field(DataType::Int32, true));
        let item: ArrayRef = Arc::new(Int32Array::from(vec![20, 30, 100]));
        let pos = Int32Array::from(vec![-1, 1, -3]);
        let got = array_insert(&sliced, &item, &pos, false, field)?;
        let expected = list(vec![
            Some(vec![Some(4), Some(5), Some(20)]),
            None,
            Some(vec![Some(100), None, Some(1)]),
        ]);
        assert_eq!(got.to_data(), expected.to_data());
        Ok(())
    }

    /// A caller that skips coercion must get an error, not a panic, when the
    /// item type differs from the element type.
    #[test]
    fn mismatched_item_type_is_an_error() {
        let list =
            ListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![Some(1)])]);
        let list: ArrayRef = Arc::new(list);
        let pos: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let item: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![2]));
        let args = vec![
            ColumnarValue::Array(Arc::clone(&list)),
            ColumnarValue::Array(pos),
            ColumnarValue::Array(item),
        ];
        let arg_fields = args
            .iter()
            .map(|arg| Arc::new(Field::new("a", arg.data_type(), true)))
            .collect();
        let error = SparkArrayInsert::new()
            .invoke_with_args(ScalarFunctionArgs {
                args,
                arg_fields,
                number_rows: 1,
                return_field: Arc::new(Field::new("r", list.data_type().clone(), true)),
                config_options: Arc::new(Default::default()),
            })
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("expects the item type Int64 to match the element type Int32"),
            "{error}"
        );
    }

    #[test]
    fn coerce_types_rules() -> Result<()> {
        let f = SparkArrayInsert::new();
        let list_of =
            |t: DataType| DataType::List(Arc::new(Field::new_list_field(t, true)));
        // smaller integer index widens to INT, item widens to the common numeric type
        let got =
            f.coerce_types(&[list_of(DataType::Int32), DataType::Int8, DataType::Int64])?;
        assert_eq!(
            got,
            vec![list_of(DataType::Int64), DataType::Int32, DataType::Int64]
        );
        // NULL item takes the element type
        let got =
            f.coerce_types(&[list_of(DataType::Utf8), DataType::Int32, DataType::Null])?;
        assert_eq!(got[2], DataType::Utf8);
        // Utf8View / LargeUtf8 elements accept a Utf8 item
        for elem in [DataType::Utf8View, DataType::LargeUtf8] {
            let got = f.coerce_types(&[
                list_of(elem.clone()),
                DataType::Int64,
                DataType::Utf8,
            ])?;
            assert_eq!(got, vec![list_of(elem.clone()), DataType::Int32, elem]);
        }
        // timestamps of different units share a common type
        let got = f.coerce_types(&[
            list_of(DataType::Timestamp(TimeUnit::Microsecond, None)),
            DataType::Int32,
            DataType::Timestamp(TimeUnit::Second, None),
        ])?;
        assert!(matches!(got[2], DataType::Timestamp(_, None)));
        assert_eq!(got[0], list_of(got[2].clone()));
        // nested types that differ only in nullability
        let inner_nullable = list_of(DataType::Int32);
        let inner_non_null =
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, false)));
        let got =
            f.coerce_types(&[list_of(inner_nullable), DataType::Int32, inner_non_null])?;
        assert_eq!(got[0], list_of(got[2].clone()));
        // Pairs with no tightest common type in Spark fail with
        // ARRAY_FUNCTION_DIFF_TYPES: a string item with numeric or nested
        // numeric elements, string with binary, and the decimal rules.
        let diff_types = |elem: DataType, item: DataType| {
            let err = f
                .coerce_types(&[list_of(elem), DataType::Int32, item])
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("[DATATYPE_MISMATCH.ARRAY_FUNCTION_DIFF_TYPES]"),
                "{err}"
            );
            err
        };
        let err = diff_types(DataType::Int32, DataType::Utf8);
        assert!(
            err.contains(r#"but it's ["ARRAY<INT>", "STRING"]."#),
            "{err}"
        );
        diff_types(list_of(DataType::Int32), list_of(DataType::Utf8));
        diff_types(DataType::Binary, DataType::Utf8);
        diff_types(DataType::Decimal128(2, 1), DataType::Float64);
        diff_types(DataType::Float32, DataType::Decimal128(5, 1));
        diff_types(DataType::Decimal128(5, 1), DataType::Decimal128(5, 2));
        diff_types(DataType::Decimal128(5, 1), DataType::Decimal128(6, 1));
        diff_types(DataType::Int64, DataType::Decimal128(5, 1));
        diff_types(DataType::Int64, DataType::Decimal128(20, 1));
        diff_types(DataType::Decimal128(9, 0), DataType::Int32);
        diff_types(
            list_of(DataType::Decimal128(5, 1)),
            list_of(DataType::Decimal128(5, 2)),
        );
        // ... and inside a map's key or value
        let map_of = |k: DataType, v: DataType| {
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![Field::new("key", k, false), Field::new("value", v, true)]
                            .into(),
                    ),
                    false,
                )),
                false,
            )
        };
        diff_types(
            map_of(DataType::Int32, DataType::Utf8),
            map_of(DataType::Int32, DataType::Int32),
        );
        diff_types(
            map_of(DataType::Utf8, DataType::Int32),
            map_of(DataType::Int32, DataType::Int32),
        );
        // A decimal that holds the whole integer range is the common type.
        for (elem, item, common) in [
            (
                DataType::Int64,
                DataType::Decimal128(20, 0),
                DataType::Decimal128(20, 0),
            ),
            (
                DataType::Int32,
                DataType::Decimal128(11, 1),
                DataType::Decimal128(11, 1),
            ),
            (
                DataType::Decimal128(11, 1),
                DataType::Int32,
                DataType::Decimal128(11, 1),
            ),
            (
                DataType::Decimal128(5, 1),
                DataType::Decimal128(5, 1),
                DataType::Decimal128(5, 1),
            ),
        ] {
            let got = f.coerce_types(&[list_of(elem), DataType::Int32, item])?;
            assert_eq!(got[2], common);
        }
        // A non-integer index fails with UNEXPECTED_INPUT_TYPE; a non-array
        // first argument with ARRAY_FUNCTION_DIFF_TYPES.
        let err = f
            .coerce_types(&[list_of(DataType::Int32), DataType::Utf8, DataType::Int32])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(r#"[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The second parameter of `array_insert` requires the "INT" type, however it has the type "STRING"."#),
            "{err}"
        );
        let err = f
            .coerce_types(&[DataType::Int32, DataType::Int32, DataType::Int32])
            .unwrap_err()
            .to_string();
        assert!(err.contains(r#"but it's ["INT", "INT"]."#), "{err}");
        Ok(())
    }
}
