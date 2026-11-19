// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Parses Spark type names with Spark's SQL type grammar, the `dataType` rule
//! of
//! [`SqlBaseParser.g4`](https://github.com/apache/spark/blob/v4.2.0/sql/api/src/main/antlr4/org/apache/spark/sql/catalyst/parser/SqlBaseParser.g4#L1477-L1525).

use std::sync::Arc;

use arrow::datatypes::{DECIMAL128_MAX_PRECISION, DataType, Field, Fields, TimeUnit};
use datafusion_common::{DataFusionError, Result, plan_datafusion_err, plan_err};

/// Parses a Spark type name, such as `int`, `decimal(10, 2)` or
/// `struct<a: int, b: array<string>>`, into the Arrow type that represents it.
///
/// Keywords are case-insensitive, as in Spark. `TIMESTAMP` is
/// `TIMESTAMP_LTZ`, Spark's default `spark.sql.timestampType`, and is tagged
/// with `session_timezone`. `CHAR(n)` and `VARCHAR(n)` are `STRING`, as Spark
/// casts them by default. Types Spark can name but this crate does not support,
/// such as intervals, `TIME`, `VARIANT` and collated strings, are an error.
pub(crate) fn parse_spark_type(
    type_name: &str,
    session_timezone: &str,
) -> Result<DataType> {
    let mut parser = Parser {
        tokens: tokenize(type_name)?,
        position: 0,
        type_name,
        session_timezone,
    };
    let data_type = parser.data_type()?;
    match parser.peek() {
        None => Ok(data_type),
        Some(token) => Err(parser.error(&format!("unexpected {token}"))),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    /// An unquoted identifier or keyword.
    Word(String),
    /// A backquoted identifier, with its quotes removed.
    Quoted(String),
    /// A single-quoted string, as in `COMMENT 'text'`.
    Text,
    Integer(u64),
    Symbol(char),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Word(word) => write!(f, "'{word}'"),
            Token::Quoted(name) => write!(f, "`{name}`"),
            Token::Text => write!(f, "string literal"),
            Token::Integer(value) => write!(f, "'{value}'"),
            Token::Symbol(symbol) => write!(f, "'{symbol}'"),
        }
    }
}

fn invalid(type_name: &str, message: &str) -> DataFusionError {
    plan_datafusion_err!("Invalid spark_cast target type '{type_name}': {message}")
}

fn tokenize(type_name: &str) -> Result<Vec<Token>> {
    let error = |message: &str| invalid(type_name, message);
    let mut tokens = vec![];
    let mut chars = type_name.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_ascii_alphabetic() || c == '_' {
            let mut word = String::new();
            while let Some(&c) = chars.peek()
                && (c.is_ascii_alphanumeric() || c == '_')
            {
                word.push(c);
                chars.next();
            }
            tokens.push(Token::Word(word));
        } else if c.is_ascii_digit() {
            let mut digits = String::new();
            while let Some(&c) = chars.peek()
                && c.is_ascii_digit()
            {
                digits.push(c);
                chars.next();
            }
            let value = digits
                .parse()
                .map_err(|_| error(&format!("'{digits}' is out of range")))?;
            tokens.push(Token::Integer(value));
        } else if c == '`' || c == '\'' {
            // A doubled quote inside a quoted identifier or string is a literal
            // quote.
            chars.next();
            let mut text = String::new();
            loop {
                match chars.next() {
                    Some(q) if q == c => {
                        if chars.peek() == Some(&c) {
                            text.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    Some(other) => text.push(other),
                    None => return Err(error("unterminated quote")),
                }
            }
            tokens.push(if c == '`' {
                Token::Quoted(text)
            } else {
                Token::Text
            });
        } else if "<>(),:".contains(c) {
            tokens.push(Token::Symbol(c));
            chars.next();
        } else {
            return Err(error(&format!("unexpected character '{c}'")));
        }
    }
    Ok(tokens)
}

const TIMESTAMP_NTZ: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);

struct Parser<'a> {
    tokens: Vec<Token>,
    position: usize,
    type_name: &'a str,
    session_timezone: &'a str,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> DataFusionError {
        invalid(self.type_name, message)
    }

    fn unsupported(&self, what: &str) -> DataFusionError {
        plan_datafusion_err!(
            "Unsupported spark_cast target type '{}': {what} is not supported",
            self.type_name
        )
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.position)
    }

    /// Applies `f` to the next token, and consumes it if `f` returns `Some`.
    fn eat_map<T>(&mut self, f: impl FnOnce(&Token) -> Option<T>) -> Option<T> {
        let value = self.peek().and_then(f);
        if value.is_some() {
            self.position += 1;
        }
        value
    }

    fn peek_keyword(&self, keyword: &str) -> bool {
        matches!(self.peek(), Some(Token::Word(word)) if word.eq_ignore_ascii_case(keyword))
    }

    fn eat_keyword(&mut self, keyword: &str) -> bool {
        let found = self.peek_keyword(keyword);
        if found {
            self.position += 1;
        }
        found
    }

    fn expect_keyword(&mut self, keyword: &str) -> Result<()> {
        if self.eat_keyword(keyword) {
            Ok(())
        } else {
            Err(self.expected(&format!("'{keyword}'")))
        }
    }

    fn eat_symbol(&mut self, symbol: char) -> bool {
        let found = self.peek() == Some(&Token::Symbol(symbol));
        if found {
            self.position += 1;
        }
        found
    }

    fn expect_symbol(&mut self, symbol: char) -> Result<()> {
        if self.eat_symbol(symbol) {
            Ok(())
        } else {
            Err(self.expected(&format!("'{symbol}'")))
        }
    }

    fn expected(&self, what: &str) -> DataFusionError {
        match self.peek() {
            Some(token) => self.error(&format!("expected {what}, found {token}")),
            None => self.error(&format!("expected {what}")),
        }
    }

    fn integer(&mut self) -> Result<u64> {
        self.eat_map(|token| match token {
            Token::Integer(value) => Some(*value),
            _ => None,
        })
        .ok_or_else(|| self.expected("an integer"))
    }

    /// The parenthesized integers after a type name, if any.
    fn parameters(&mut self) -> Result<Vec<u64>> {
        let mut parameters = vec![];
        if self.eat_symbol('(') {
            loop {
                parameters.push(self.integer()?);
                if !self.eat_symbol(',') {
                    break;
                }
            }
            self.expect_symbol(')')?;
        }
        Ok(parameters)
    }

    fn data_type(&mut self) -> Result<DataType> {
        let Some(word) = self.eat_map(|token| match token {
            Token::Word(word) => Some(word.to_ascii_lowercase()),
            _ => None,
        }) else {
            return Err(self.expected("a type"));
        };
        let data_type = match word.as_str() {
            "array" => {
                self.expect_symbol('<')?;
                let element = self.data_type()?;
                self.expect_symbol('>')?;
                DataType::List(Arc::new(Field::new_list_field(element, true)))
            }
            "map" => {
                self.expect_symbol('<')?;
                let key = self.data_type()?;
                self.expect_symbol(',')?;
                let value = self.data_type()?;
                self.expect_symbol('>')?;
                // The field names Spark's `ArrowUtils` uses for a map.
                let entries = Field::new_struct(
                    "entries",
                    vec![
                        Field::new("key", key, false),
                        Field::new("value", value, true),
                    ],
                    false,
                );
                DataType::Map(Arc::new(entries), false)
            }
            "struct" => DataType::Struct(self.struct_fields()?),
            "boolean" => DataType::Boolean,
            "tinyint" | "byte" => DataType::Int8,
            "smallint" | "short" => DataType::Int16,
            "int" | "integer" => DataType::Int32,
            "bigint" | "long" => DataType::Int64,
            "float" | "real" => DataType::Float32,
            "double" => DataType::Float64,
            "date" => DataType::Date32,
            "binary" => DataType::Binary,
            "timestamp_ltz" => self.timestamp_ltz(),
            "timestamp_ntz" => TIMESTAMP_NTZ,
            "timestamp" => {
                if self.eat_keyword("with") {
                    for keyword in ["local", "time", "zone"] {
                        self.expect_keyword(keyword)?;
                    }
                    self.timestamp_ltz()
                } else if self.eat_keyword("without") {
                    for keyword in ["time", "zone"] {
                        self.expect_keyword(keyword)?;
                    }
                    TIMESTAMP_NTZ
                } else {
                    self.timestamp_ltz()
                }
            }
            "string" => DataType::Utf8,
            "char" | "character" | "varchar" => {
                match self.parameters()?.len() {
                    1 => DataType::Utf8,
                    // Spark requires a length.
                    0 => {
                        let name = word.to_ascii_uppercase();
                        return plan_err!(
                            "[DATATYPE_MISSING_SIZE] DataType \"{name}\" requires a length \
                             parameter, for example \"{name}\"(10). Please specify the length. \
                             SQLSTATE: 42K01"
                        );
                    }
                    _ => return Err(self.error(&format!("{word} takes one length"))),
                }
            }
            "decimal" | "dec" | "numeric" => self.decimal()?,
            "interval" | "time" | "variant" | "void" | "geography" | "geometry" => {
                return Err(self.unsupported(&word.to_ascii_uppercase()));
            }
            _ => return Err(self.error(&format!("unknown type '{word}'"))),
        };
        if self.peek_keyword("collate") {
            return Err(self.unsupported("a collation"));
        }
        Ok(data_type)
    }

    fn timestamp_ltz(&self) -> DataType {
        DataType::Timestamp(TimeUnit::Microsecond, Some(self.session_timezone.into()))
    }

    fn decimal(&mut self) -> Result<DataType> {
        let (precision, scale) = match self.parameters()?.as_slice() {
            // Spark's default for a decimal without a precision.
            [] => (10, 0),
            [precision] => (*precision, 0),
            [precision, scale] => (*precision, *scale),
            _ => return Err(self.error("decimal takes a precision and a scale")),
        };
        if !(1..=u64::from(DECIMAL128_MAX_PRECISION)).contains(&precision) {
            return Err(self.error(&format!(
                "decimal precision {precision} is not between 1 and \
                 {DECIMAL128_MAX_PRECISION}"
            )));
        }
        if scale > precision {
            return Err(self.error(&format!(
                "decimal scale {scale} is greater than its precision {precision}"
            )));
        }
        // Both fit, since precision is at most 38.
        Ok(DataType::Decimal128(precision as u8, scale as i8))
    }

    /// `struct<>`, or `struct<name[:] type [NOT NULL] [COMMENT '...'], ...>`.
    fn struct_fields(&mut self) -> Result<Fields> {
        self.expect_symbol('<')?;
        let mut fields = vec![];
        if self.eat_symbol('>') {
            return Ok(fields.into());
        }
        loop {
            let Some(name) = self.eat_map(|token| match token {
                Token::Word(name) | Token::Quoted(name) => Some(name.clone()),
                _ => None,
            }) else {
                return Err(self.expected("a field name"));
            };
            self.eat_symbol(':');
            let data_type = self.data_type()?;
            let nullable = if self.eat_keyword("not") {
                self.expect_keyword("null")?;
                false
            } else {
                true
            };
            if self.eat_keyword("comment")
                && self
                    .eat_map(|token| (token == &Token::Text).then_some(()))
                    .is_none()
            {
                return Err(self.expected("a comment string"));
            }
            fields.push(Field::new(name, data_type, nullable));
            if !self.eat_symbol(',') {
                break;
            }
        }
        self.expect_symbol('>')?;
        Ok(fields.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(type_name: &str) -> DataType {
        parse_spark_type(type_name, "America/New_York").unwrap()
    }

    fn parse_error(type_name: &str) -> String {
        parse_spark_type(type_name, "UTC").unwrap_err().to_string()
    }

    fn list(element: DataType) -> DataType {
        DataType::List(Arc::new(Field::new_list_field(element, true)))
    }

    #[test]
    fn test_primitive_types() {
        let ltz =
            DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into()));
        let ntz = DataType::Timestamp(TimeUnit::Microsecond, None);
        for (type_name, expected) in [
            ("boolean", DataType::Boolean),
            ("tinyint", DataType::Int8),
            ("byte", DataType::Int8),
            ("smallint", DataType::Int16),
            ("short", DataType::Int16),
            ("int", DataType::Int32),
            ("INTEGER", DataType::Int32),
            ("bigint", DataType::Int64),
            ("Long", DataType::Int64),
            ("float", DataType::Float32),
            ("real", DataType::Float32),
            ("double", DataType::Float64),
            ("date", DataType::Date32),
            ("binary", DataType::Binary),
            ("string", DataType::Utf8),
            ("char(3)", DataType::Utf8),
            ("varchar(10)", DataType::Utf8),
            ("timestamp", ltz.clone()),
            ("timestamp_ltz", ltz.clone()),
            ("timestamp with local time zone", ltz),
            ("timestamp_ntz", ntz.clone()),
            ("TIMESTAMP WITHOUT TIME ZONE", ntz),
            ("decimal", DataType::Decimal128(10, 0)),
            ("dec(5)", DataType::Decimal128(5, 0)),
            ("numeric(38, 38)", DataType::Decimal128(38, 38)),
            ("  decimal ( 10 , 2 )  ", DataType::Decimal128(10, 2)),
        ] {
            assert_eq!(parse(type_name), expected, "{type_name}");
        }
    }

    #[test]
    fn test_complex_types() {
        assert_eq!(parse("array<int>"), list(DataType::Int32));
        assert_eq!(parse("ARRAY<ARRAY<string>>"), list(list(DataType::Utf8)));

        let DataType::Map(entries, false) = parse("map<string, array<int>>") else {
            panic!("expected a map");
        };
        let DataType::Struct(fields) = entries.data_type() else {
            panic!("expected struct entries");
        };
        assert_eq!(
            fields[0].as_ref(),
            &Field::new("key", DataType::Utf8, false)
        );
        assert_eq!(
            fields[1].as_ref(),
            &Field::new("value", list(DataType::Int32), true)
        );

        assert_eq!(
            parse(
                "struct<a: int, b string NOT NULL COMMENT 'it''s b', `c d`: \
                 struct<a:decimal(4,1)>, a bigint>"
            ),
            DataType::Struct(Fields::from(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, false),
                Field::new(
                    "c d",
                    DataType::Struct(Fields::from(vec![Field::new(
                        "a",
                        DataType::Decimal128(4, 1),
                        true
                    )])),
                    true
                ),
                // Spark allows duplicate field names.
                Field::new("a", DataType::Int64, true),
            ]))
        );
        assert_eq!(parse("struct<>"), DataType::Struct(Fields::empty()));
    }

    /// The canonical Spark name of each type parses to a type that
    /// [`spark_sql_type`] names the same way.
    #[test]
    fn test_round_trips_with_spark_sql_type() {
        use crate::function::type_coercion::spark_sql_type;
        for type_name in [
            "BOOLEAN",
            "TINYINT",
            "SMALLINT",
            "INT",
            "BIGINT",
            "FLOAT",
            "DOUBLE",
            "DECIMAL(10,0)",
            "DECIMAL(38,18)",
            "STRING",
            "BINARY",
            "DATE",
            "TIMESTAMP",
            "TIMESTAMP_NTZ",
            "ARRAY<ARRAY<INT>>",
            "MAP<STRING, ARRAY<TIMESTAMP>>",
            "STRUCT<a: INT, `c d`: STRING NOT NULL, s: STRUCT<x: DATE>>",
            "STRUCT<>",
        ] {
            assert_eq!(spark_sql_type(&parse(type_name)), type_name);
        }
    }

    /// Spark requires a length for CHAR and VARCHAR.
    #[test]
    fn test_char_and_varchar_require_a_length() {
        for (type_name, name) in [
            ("char", "CHAR"),
            ("Character", "CHARACTER"),
            ("varchar", "VARCHAR"),
        ] {
            assert_eq!(
                parse_error(type_name),
                format!(
                    "Error during planning: [DATATYPE_MISSING_SIZE] DataType \"{name}\" requires a \
                     length parameter, for example \"{name}\"(10). Please specify the length. \
                     SQLSTATE: 42K01"
                )
            );
        }
    }

    #[test]
    fn test_invalid_types() {
        for (type_name, message) in [
            ("", "expected a type"),
            ("integer2", "unknown type 'integer2'"),
            ("int int", "unexpected 'int'"),
            ("array", "expected '<'"),
            ("array<int", "expected '>'"),
            ("map<int>", "expected ','"),
            ("struct<a>", "expected a type, found '>'"),
            ("struct<a int,>", "expected a field name, found '>'"),
            ("struct<`a int>", "unterminated quote"),
            ("decimal(0)", "precision 0 is not between 1 and 38"),
            ("decimal(39)", "precision 39 is not between 1 and 38"),
            ("decimal(5, 6)", "scale 6 is greater than its precision 5"),
            ("decimal(1, 2, 3)", "decimal takes a precision and a scale"),
            ("decimal(-1)", "unexpected character '-'"),
            ("timestamp with time zone", "expected 'local', found 'time'"),
        ] {
            let error = parse_error(type_name);
            assert!(
                error.contains("Invalid spark_cast target type")
                    && error.contains(message),
                "{type_name}: {error}"
            );
        }
    }

    #[test]
    fn test_unsupported_types() {
        for type_name in [
            "interval",
            "interval day to second",
            "time",
            "variant",
            "void",
            "string collate UTF8_LCASE",
            "array<interval year>",
        ] {
            let error = parse_error(type_name);
            assert!(
                error.contains("Unsupported spark_cast target type"),
                "{type_name}: {error}"
            );
        }
    }
}
