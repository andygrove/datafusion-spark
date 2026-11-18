// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Typed Spark errors.
//!
//! A [`SparkError`] renders the message Spark raises for the same condition,
//! including its error class (such as `[CAST_OVERFLOW]`). It reaches callers as
//! [`DataFusionError::External`], so an engine that maps errors back to Spark
//! exceptions, such as Comet, can recover the variant with
//! [`SparkError::find`] instead of parsing the message.

use std::error::Error;
use std::fmt::{Display, Formatter};

use datafusion_common::DataFusionError;

/// An error that Spark raises with a specific error class.
///
/// The messages follow Spark 4.x's
/// [`error-conditions.json`](https://github.com/apache/spark/blob/v4.2.0/common/utils/src/main/resources/error/error-conditions.json).
/// Spark 3.x appended `If necessary set "spark.sql.ansi.enabled" to "false" to
/// bypass this error.` to the `CAST_INVALID_INPUT` and `CAST_OVERFLOW`
/// messages, and 4.0 dropped it.
/// Spark quotes values and types in the forms its `toSQLValue` and `toSQLType`
/// produce. The caller passes them already formatted, such as `2147483648L`,
/// `NaN`, or `'abc'` with its quotes, and `"BIGINT"` without the quotes, since
/// only the caller knows the Spark type. Every message ends with the error's
/// SQLSTATE, as Spark's do.
///
/// The variants are `#[non_exhaustive]` so that fields can be added later.
/// Code outside this crate matches them with `..`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparkError {
    /// `CAST_INVALID_INPUT`: a value that cannot be parsed as the target type,
    /// in ANSI mode. Spark raises it as a `SparkNumberFormatException`.
    #[non_exhaustive]
    CastInvalidValue {
        value: String,
        from_type: String,
        to_type: String,
    },
    /// The same error class and message as [`SparkError::CastInvalidValue`],
    /// for a cast to a date or timestamp. Spark raises it as a
    /// `SparkDateTimeException` rather than a `SparkNumberFormatException`.
    #[non_exhaustive]
    InvalidInputInCastToDatetime {
        value: String,
        from_type: String,
        to_type: String,
    },
    /// `CAST_OVERFLOW`: a value outside the range of the target type, in ANSI
    /// mode.
    #[non_exhaustive]
    CastOverflow {
        value: String,
        from_type: String,
        to_type: String,
    },
    /// `NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION`: a value that does not fit
    /// the target decimal precision and scale, in ANSI mode.
    #[non_exhaustive]
    NumericValueOutOfRange {
        value: String,
        precision: u8,
        scale: i8,
    },
    /// `NUMERIC_OUT_OF_SUPPORTED_RANGE`: a string with more than 38 digits cast
    /// to a decimal.
    #[non_exhaustive]
    NumericOutOfSupportedRange { value: String },
}

impl SparkError {
    /// The Spark error class, such as `CAST_OVERFLOW`.
    pub fn error_class(&self) -> &'static str {
        match self {
            Self::CastInvalidValue { .. } | Self::InvalidInputInCastToDatetime { .. } => {
                "CAST_INVALID_INPUT"
            }
            Self::CastOverflow { .. } => "CAST_OVERFLOW",
            Self::NumericValueOutOfRange { .. } => {
                "NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION"
            }
            Self::NumericOutOfSupportedRange { .. } => "NUMERIC_OUT_OF_SUPPORTED_RANGE",
        }
    }

    /// The SQLSTATE that Spark reports with the error class.
    pub fn sql_state(&self) -> &'static str {
        match self {
            Self::CastInvalidValue { .. } | Self::InvalidInputInCastToDatetime { .. } => {
                "22018"
            }
            Self::CastOverflow { .. }
            | Self::NumericValueOutOfRange { .. }
            | Self::NumericOutOfSupportedRange { .. } => "22003",
        }
    }

    /// Finds a `SparkError` in `error`, looking through the
    /// [`DataFusionError::Context`], [`DataFusionError::Shared`] and
    /// [`DataFusionError::ArrowError`] wrappers that DataFusion and Arrow
    /// kernels add around an external error.
    pub fn find(error: &DataFusionError) -> Option<&SparkError> {
        match error {
            DataFusionError::External(e) => e.downcast_ref::<SparkError>(),
            DataFusionError::ArrowError(e, _) => match e.as_ref() {
                arrow::error::ArrowError::ExternalError(e) => e.downcast_ref(),
                _ => None,
            },
            DataFusionError::Context(_, e) => Self::find(e),
            DataFusionError::Shared(e) => Self::find(e),
            _ => None,
        }
    }
}

impl Display for SparkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let class = self.error_class();
        let sql_state = self.sql_state();
        match self {
            Self::CastInvalidValue {
                value,
                from_type,
                to_type,
            }
            | Self::InvalidInputInCastToDatetime {
                value,
                from_type,
                to_type,
            } => write!(
                f,
                "[{class}] The value {value} of the type \"{from_type}\" cannot be cast to \
                 \"{to_type}\" because it is malformed. Correct the value as per the syntax, or \
                 change its target type. Use `try_cast` to tolerate malformed input and return \
                 NULL instead. SQLSTATE: {sql_state}"
            ),
            Self::CastOverflow {
                value,
                from_type,
                to_type,
            } => write!(
                f,
                "[{class}] The value {value} of the type \"{from_type}\" cannot be cast to \
                 \"{to_type}\" due to an overflow. Use `try_cast` to tolerate overflow and return \
                 NULL instead. SQLSTATE: {sql_state}"
            ),
            Self::NumericValueOutOfRange {
                value,
                precision,
                scale,
            } => write!(
                f,
                // Spark renders the empty parent template, a space, then the
                // subclass template, so there are two spaces after the class.
                "[{class}]  {value} cannot be represented as Decimal({precision}, {scale}). If \
                 necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass this error, and \
                 return NULL instead. SQLSTATE: {sql_state}"
            ),
            Self::NumericOutOfSupportedRange { value } => write!(
                f,
                "[{class}] The value {value} cannot be interpreted as a numeric since it has \
                 more than 38 digits. SQLSTATE: {sql_state}"
            ),
        }
    }
}

impl Error for SparkError {}

impl From<SparkError> for DataFusionError {
    fn from(error: SparkError) -> Self {
        DataFusionError::External(Box::new(error))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::error::ArrowError;
    use std::sync::Arc;

    fn overflow() -> SparkError {
        SparkError::CastOverflow {
            value: "2147483648L".to_string(),
            from_type: "BIGINT".to_string(),
            to_type: "INT".to_string(),
        }
    }

    // Messages from Spark 4.2.0's error-conditions.json.
    #[test]
    fn test_messages_match_spark() {
        assert_eq!(
            overflow().to_string(),
            "[CAST_OVERFLOW] The value 2147483648L of the type \"BIGINT\" cannot be cast to \
             \"INT\" due to an overflow. Use `try_cast` to tolerate overflow and return NULL \
             instead. SQLSTATE: 22003"
        );
        let invalid = SparkError::CastInvalidValue {
            value: "'abc'".to_string(),
            from_type: "STRING".to_string(),
            to_type: "INT".to_string(),
        };
        assert_eq!(
            invalid.to_string(),
            "[CAST_INVALID_INPUT] The value 'abc' of the type \"STRING\" cannot be cast to \
             \"INT\" because it is malformed. Correct the value as per the syntax, or change its \
             target type. Use `try_cast` to tolerate malformed input and return NULL instead. \
             SQLSTATE: 22018"
        );
        let datetime = SparkError::InvalidInputInCastToDatetime {
            value: "'abc'".to_string(),
            from_type: "STRING".to_string(),
            to_type: "DATE".to_string(),
        };
        assert!(datetime.to_string().starts_with(
            "[CAST_INVALID_INPUT] The value 'abc' of the type \"STRING\" cannot be cast to \"DATE\""
        ));
        let out_of_range = SparkError::NumericValueOutOfRange {
            value: "123.45".to_string(),
            precision: 4,
            scale: 2,
        };
        assert_eq!(
            out_of_range.to_string(),
            "[NUMERIC_VALUE_OUT_OF_RANGE.WITH_SUGGESTION]  123.45 cannot be represented as \
             Decimal(4, 2). If necessary set \"spark.sql.ansi.enabled\" to \"false\" to bypass \
             this error, and return NULL instead. SQLSTATE: 22003"
        );
        let too_many_digits = SparkError::NumericOutOfSupportedRange {
            value: "1".repeat(39),
        };
        assert_eq!(
            too_many_digits.to_string(),
            format!(
                "[NUMERIC_OUT_OF_SUPPORTED_RANGE] The value {} cannot be interpreted as a \
                 numeric since it has more than 38 digits. SQLSTATE: 22003",
                "1".repeat(39)
            )
        );
    }

    /// The message uses the value as given, so a double that Spark's
    /// `toSQLValue` leaves unquoted, such as `NaN`, stays unquoted.
    #[test]
    fn test_values_are_formatted_by_the_caller() {
        let nan = SparkError::InvalidInputInCastToDatetime {
            value: "NaN".to_string(),
            from_type: "DOUBLE".to_string(),
            to_type: "TIMESTAMP".to_string(),
        };
        assert!(
            nan.to_string()
                .starts_with("[CAST_INVALID_INPUT] The value NaN of the type \"DOUBLE\""),
            "{nan}"
        );
    }

    #[test]
    fn test_find_through_wrappers() {
        let external = DataFusionError::from(overflow());
        assert_eq!(SparkError::find(&external), Some(&overflow()));
        assert!(external.to_string().contains(&overflow().to_string()));

        let context = external.context("while evaluating");
        assert_eq!(SparkError::find(&context), Some(&overflow()));

        let shared = DataFusionError::Shared(Arc::new(context));
        assert_eq!(SparkError::find(&shared), Some(&overflow()));

        // Arrow's `try_unary` kernels can only return an `ArrowError`.
        let arrow =
            DataFusionError::from(ArrowError::ExternalError(Box::new(overflow())));
        assert_eq!(SparkError::find(&arrow), Some(&overflow()));

        assert_eq!(
            SparkError::find(&DataFusionError::Execution("x".to_string())),
            None
        );
    }
}
