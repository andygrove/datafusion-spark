#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Run the datafusion-spark sqllogictest files against Apache Spark 4.2.

Each record in crates/sqllogictest/test_files/**/*.slt is run on a
local PySpark session, and Spark's result is compared with the expected result
in the file. The expected results come from DataFusion, so every difference
reported here is a query where DataFusion and Spark disagree, or where the
test expects the wrong value.

Running Spark is manual; only the unit tests run in CI. See README.md for setup.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
import re
import struct
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Optional

SPARK_VERSION = "4.2"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_TEST_DIR = REPO_ROOT / "crates" / "sqllogictest" / "test_files"

# The label that `skipif` and `onlyif` directives use for this runner.
LABEL = "spark"

# DataFusion runs these files with ANSI mode off and no time zone (UTC), so
# every file starts from the same Spark settings.
SPARK_DEFAULTS = {
    "spark.sql.ansi.enabled": "false",
    "spark.sql.session.timeZone": "UTC",
    "spark.sql.mapKeyDedupPolicy": "EXCEPTION",
    # Spark 4.2 rejects the TIME type unless this internal setting is on.
    "spark.sql.timeType.enabled": "true",
}

# DataFusion settings that have a Spark equivalent.
SETTINGS = {
    "datafusion.execution.enable_ansi_mode": "spark.sql.ansi.enabled",
    "datafusion.execution.time_zone": "spark.sql.session.timeZone",
    "datafusion.spark.map_key_dedup_policy": "spark.sql.mapKeyDedupPolicy",
}

# Type names DataFusion accepts and Spark does not.
TYPE_ALIASES = {
    "BYTEA": "BINARY",
    "FLOAT4": "FLOAT",
    "FLOAT8": "DOUBLE",
    "INT2": "SMALLINT",
    "INT4": "INT",
    "INT8": "BIGINT",
    "TEXT": "STRING",
}

# Spark's maximum decimal precision.
MAX_DECIMAL_PRECISION = 38

# The most rows collected from Spark for one query. Some tests expect an error
# for a query that Spark answers with billions of rows, such as stack(2000000000, 1).
MAX_ROWS = 1000

# The most result lines printed for one difference.
MAX_REPORT_LINES = 20

# The skip reasons the summary lists without --verbose.
MAX_SKIP_REASONS = 15


class Skip(Exception):
    """The record cannot be run on Spark. The message says why."""


class Unsupported(Exception):
    """Spark returned a value of a type this runner does not format."""


# ---------------------------------------------------------------------------
# Parsing .slt files
# ---------------------------------------------------------------------------


@dataclass
class Record:
    """A `statement` or `query` record from an .slt file."""

    path: Path
    line: int
    kind: str  # "statement" or "query"
    sql: str = ""
    expect_error: bool = False
    types: str = ""
    sort_mode: str = "nosort"
    expected: list[str] = field(default_factory=list)
    skip_reason: Optional[str] = None

    @property
    def location(self) -> str:
        return f"{_display_path(self.path)}:{self.line}"


def parse_slt(path: Path) -> list[Record]:
    """Parse the `statement` and `query` records of an .slt file."""
    lines = path.read_text().split("\n")
    records: list[Record] = []
    conditions: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        words = lines[i].split()
        if not words or words[0].startswith("#"):
            i += 1
            continue
        if words[0] in ("skipif", "onlyif") and len(words) > 1:
            conditions.append((words[0], words[1]))
            i += 1
            continue
        if words[0] not in ("statement", "query"):
            # Directives such as `include` and `halt`, which these files do not use.
            conditions = []
            i += 1
            continue

        record = Record(path=path, line=i + 1, kind=words[0])
        args = words[1:]
        if args and args[0] == "error":
            record.expect_error = True
        elif record.kind == "query" and args:
            record.types = args[0]
            if len(args) > 1 and args[1] in ("nosort", "rowsort", "valuesort"):
                record.sort_mode = args[1]
        for directive, label in conditions:
            if directive == "skipif" and label == LABEL:
                record.skip_reason = f"skipif {label}"
            elif directive == "onlyif" and label != LABEL:
                record.skip_reason = f"onlyif {label}"
        conditions = []

        # The SQL runs until a blank line or the `----` separator.
        i += 1
        sql = []
        while i < len(lines) and lines[i].strip() and lines[i] != "----":
            if not lines[i].lstrip().startswith("#"):
                sql.append(lines[i])
            i += 1
        record.sql = "\n".join(sql)

        # Expected results (or an expected error message) run until an empty
        # line. A result line may start with `#`, so it is not a comment here,
        # and it may be a single space, the engine's empty first row before a
        # multi-line value.
        if i < len(lines) and lines[i] == "----":
            i += 1
            while i < len(lines) and lines[i] != "":
                record.expected.append(lines[i])
                i += 1
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Translating DataFusion SQL to Spark SQL
#
# Spark 4.2 parses most of these files as written, including `::` casts.
# Only a few renames are applied here, and `arrow_cast` and `spark_cast`
# become a CAST (see below). Anything else Spark cannot parse, and any function
# Spark does not have (such as `arrow_typeof`), is skipped.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
      (?P<string>'(?:[^']|'')*')
    | (?P<quoted>`[^`]*`|"[^"]*")
    | (?P<comment>--[^\n]*)
    | (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)
    | (?P<cast>::)
    | (?P<space>\s+)
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)

_COLUMN_RE = re.compile(r"column(\d+)", re.IGNORECASE)


def _tokenize(sql: str) -> list[tuple[str, str]]:
    return [(m.lastgroup, m.group()) for m in _TOKEN_RE.finditer(sql)]


def _next_significant(tokens: list[tuple[str, str]], start: int) -> int:
    """Return the index of the next token that is not whitespace or a comment."""
    i = start
    while i < len(tokens) and tokens[i][0] in ("space", "comment"):
        i += 1
    return i


def _translate_type(tokens: list[tuple[str, str]], i: int) -> str:
    """Translate the type name at tokens[i]."""
    text = tokens[i][1]
    upper = text.upper()
    if upper in TYPE_ALIASES:
        return TYPE_ALIASES[upper]
    j = _next_significant(tokens, i + 1)
    has_args = j < len(tokens) and tokens[j][1] == "("
    if upper == "VARCHAR" and not has_args:
        return "STRING"
    if upper in ("DECIMAL", "DEC", "NUMERIC") and has_args:
        k = _next_significant(tokens, j + 1)
        if k < len(tokens) and tokens[k][0] == "number":
            if int(tokens[k][1]) > MAX_DECIMAL_PRECISION:
                raise Skip(f"DECIMAL precision above {MAX_DECIMAL_PRECISION}")
    return text


# ---------------------------------------------------------------------------
# arrow_cast
#
# `arrow_cast(expr, 'Type')` casts to an Arrow type. Many Arrow types differ
# from a Spark type only in their physical layout: Utf8, LargeUtf8 and Utf8View
# are all Spark's STRING, and a dictionary is an encoding of its value type. So
# the call becomes a CAST to that Spark type. A type Spark does not have, such
# as an unsigned integer, skips the record.
#
# Arrow interprets a number cast to a date, time or timestamp as a count of the
# type's unit (days, seconds, nanoseconds, ...), and Spark does not, so those
# casts are only translated for literal arguments.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrowType:
    """A parsed Arrow type, such as `Decimal128(10, 2)` or `List(Int32)`."""

    name: str
    args: tuple = ()


@dataclass(frozen=True)
class ArrowField:
    """A named field in a `Struct` or `Map` type, such as `"a": Int64`."""

    name: str
    type: object


_ARROW_TOKEN_RE = re.compile(
    r"""\s*(?:
      (?P<string>"(?:[^"\\]|\\.)*")
    | (?P<number>-?\d+)
    | (?P<word>[A-Za-zµ_][A-Za-z0-9µ_-]*)
    | (?P<punct>[(),:])
    )""",
    re.VERBOSE,
)


def parse_arrow_type(text: str):
    """Parse an Arrow type as DataFusion's arrow_cast spells it."""
    tokens = []
    pos, text = 0, text.strip()
    while pos < len(text):
        m = _ARROW_TOKEN_RE.match(text, pos)
        if not m:
            raise ValueError(text)
        tokens.append((m.lastgroup, m.group(m.lastgroup)))
        pos = m.end()
    i = 0

    def peek() -> Optional[str]:
        return tokens[i][1] if i < len(tokens) else None

    def expect(value: str) -> None:
        nonlocal i
        if peek() != value:
            raise ValueError(text)
        i += 1

    def value():
        nonlocal i
        if i >= len(tokens):
            raise ValueError(text)
        kind, token = tokens[i]
        i += 1
        if kind == "number":
            return int(token)
        if kind == "string":
            name = token[1:-1]
            if peek() == ":":
                i += 1
                return ArrowField(name, value())
            return name
        if kind == "word":
            # Nullability does not change the Spark type.
            if token in ("non-null", "nullable"):
                return value()
            args = []
            if peek() == "(":
                i += 1
                while peek() != ")":
                    args.append(value())
                    if peek() == ",":
                        i += 1
                expect(")")
            return ArrowType(token, tuple(args))
        raise ValueError(text)

    parsed = value()
    if i != len(tokens) or not isinstance(parsed, ArrowType):
        raise ValueError(text)
    return parsed


_ARROW_SIMPLE_TYPES = {
    "Boolean": "BOOLEAN",
    "Int8": "TINYINT",
    "Int16": "SMALLINT",
    "Int32": "INT",
    "Int64": "BIGINT",
    "Float32": "FLOAT",
    "Float64": "DOUBLE",
    "Utf8": "STRING",
    "LargeUtf8": "STRING",
    "Utf8View": "STRING",
    "Binary": "BINARY",
    "LargeBinary": "BINARY",
    "BinaryView": "BINARY",
    "FixedSizeBinary": "BINARY",
    "Date32": "DATE",
    "Date64": "DATE",
}

_ARROW_LIST_TYPES = ("List", "LargeList", "ListView", "LargeListView", "FixedSizeList")

_ARROW_DECIMAL_TYPES = ("Decimal32", "Decimal64", "Decimal128", "Decimal256")

_ARROW_UNITS = {
    "Second": "s",
    "s": "s",
    "Millisecond": "ms",
    "ms": "ms",
    "Microsecond": "us",
    "µs": "us",
    "us": "us",
    "Nanosecond": "ns",
    "ns": "ns",
}

# Fractional-second digits each time unit keeps.
_UNIT_DIGITS = {"s": 0, "ms": 3, "us": 6, "ns": 9}

_UTC_NAMES = {"UTC", "ETC/UTC", "Z", "+00:00", "+00", "+0000", "GMT"}


def _is_utc(name: str) -> bool:
    return name.upper() in _UTC_NAMES


def _arrow_unit(t: ArrowType) -> str:
    if not t.args or not isinstance(t.args[0], ArrowType) or t.args[0].name not in _ARROW_UNITS:
        raise Skip("arrow_cast to a type this runner cannot parse")
    return _ARROW_UNITS[t.args[0].name]


def _arrow_time_zone(t: ArrowType) -> Optional[str]:
    """The time zone of an Arrow Timestamp: `None`, `Some("UTC")` or `"UTC"`."""
    if len(t.args) < 2:
        return None
    tz = t.args[1]
    if isinstance(tz, str):
        return tz
    if isinstance(tz, ArrowType) and tz.name == "Some" and len(tz.args) == 1:
        return tz.args[0]
    if isinstance(tz, ArrowType) and tz.name == "None":
        return None
    raise Skip("arrow_cast to a type this runner cannot parse")


def _same_time_zone(a: str, b: str) -> bool:
    return a == b or (_is_utc(a) and _is_utc(b))


def spark_type_for_arrow(t: ArrowType) -> str:
    """The Spark type with the same values as an Arrow type, or raise Skip."""
    name = t.name
    if name in _ARROW_SIMPLE_TYPES:
        return _ARROW_SIMPLE_TYPES[name]
    if name in ("UInt8", "UInt16", "UInt32", "UInt64"):
        raise Skip("arrow_cast to an unsigned integer, which Spark does not have")
    if name in _ARROW_DECIMAL_TYPES:
        precision, scale = t.args
        if precision > MAX_DECIMAL_PRECISION:
            raise Skip(f"DECIMAL precision above {MAX_DECIMAL_PRECISION}")
        if scale < 0:
            raise Skip("arrow_cast to a DECIMAL with a negative scale")
        return f"DECIMAL({precision}, {scale})"
    if name in _ARROW_LIST_TYPES:
        return f"ARRAY<{spark_type_for_arrow(t.args[-1])}>"
    if name == "Dictionary":
        return spark_type_for_arrow(t.args[1])
    if name == "Struct":
        fields = [f"`{f.name}`: {spark_type_for_arrow(f.type)}" for f in t.args]
        return f"STRUCT<{', '.join(fields)}>"
    if name == "Map":
        entries = t.args[0]
        key, value = entries.type.args
        return f"MAP<{spark_type_for_arrow(key.type)}, {spark_type_for_arrow(value.type)}>"
    if name in ("Time32", "Time64"):
        unit = _arrow_unit(t)
        if unit == "ns":
            raise Skip("arrow_cast to nanoseconds, which Spark does not have")
        return f"TIME({_UNIT_DIGITS[unit]})"
    if name == "Timestamp":
        unit = _arrow_unit(t)
        if unit != "us" or _arrow_time_zone(t) is not None:
            raise Skip("arrow_cast to a nested timestamp that is not TIMESTAMP_NTZ")
        return "TIMESTAMP_NTZ"
    raise Skip(f"arrow_cast to {name}, which Spark does not have")


def _unquote(token: tuple[str, str]) -> str:
    """The text of an SQL string literal token."""
    return token[1][1:-1].replace("''", "'")


def _quote(text: str) -> str:
    """An SQL string literal for text."""
    return "'" + text.replace("'", "''") + "'"


def _strip_parens(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop whitespace, comments and parentheses that wrap the whole argument."""
    tokens = [t for t in tokens if t[0] not in ("space", "comment")]
    while len(tokens) >= 2 and tokens[0][1] == "(" and tokens[-1][1] == ")":
        if _matching_paren(tokens, 0) != len(tokens) - 1:
            return tokens
        tokens = tokens[1:-1]
    return tokens


def _literal(tokens: list[tuple[str, str]]):
    """Return ("null", None), ("string", text) or ("integer", n) for a literal argument."""
    tokens = _strip_parens(tokens)
    if len(tokens) == 1 and tokens[0][0] == "word" and tokens[0][1].upper() == "NULL":
        return ("null", None)
    if len(tokens) == 1 and tokens[0][0] == "string":
        return ("string", _unquote(tokens[0]))
    # A typed literal, such as DATE '2024-01-01' or TIMESTAMP '2024-01-01 10:00:00'
    if (
        len(tokens) == 2
        and tokens[0][0] == "word"
        and tokens[0][1].upper() in ("DATE", "TIMESTAMP", "TIMESTAMP_NTZ")
        and tokens[1][0] == "string"
    ):
        return ("typed", _unquote(tokens[1]))
    sign = 1
    if len(tokens) == 2 and tokens[0][1] in ("-", "+"):
        sign = -1 if tokens[0][1] == "-" else 1
        tokens = tokens[1:]
    if len(tokens) == 1 and tokens[0][0] == "number" and tokens[0][1].isdigit():
        return ("integer", sign * int(tokens[0][1]))
    return (None, None)


_FRACTION_RE = re.compile(r":\d{2}\.(\d+)")

_EPOCH_FUNCTIONS = {"s": "timestamp_seconds", "ms": "timestamp_millis", "us": "timestamp_micros"}

_MILLIS_PER_DAY = 86_400_000


def _temporal_cast(t: ArrowType, arg: list[tuple[str, str]], arg_sql: str, time_zone: str) -> str:
    """Translate arrow_cast to a date, time or timestamp, which depends on the argument."""
    kind, value = _literal(arg)
    if kind == "typed":
        # DataFusion's TIMESTAMP literal has no time zone and Spark's is in the
        # session time zone, so they are the same value only in UTC.
        if t.name == "Timestamp" and not _is_utc(time_zone):
            raise Skip(f"arrow_cast of a typed literal to {t.name} in a session time zone other than UTC")
        kind = "string"
        arg_sql = _quote(value)
    if kind is None:
        raise Skip(f"arrow_cast of an expression to {t.name}, which Arrow and Spark convert differently")
    if t.name in ("Date32", "Date64"):
        if kind == "integer":
            days = value
            if t.name == "Date64":
                if value % _MILLIS_PER_DAY:
                    raise Skip("arrow_cast to Date64 of milliseconds that are not whole days")
                days = value // _MILLIS_PER_DAY
            return f"date_from_unix_date({days})"
        return f"CAST({arg_sql} AS DATE)"
    if t.name in ("Time32", "Time64"):
        spark_type = spark_type_for_arrow(t)
        if kind == "integer":
            raise Skip(f"arrow_cast of a number to {t.name}, which Spark cannot cast")
        return f"CAST({arg_sql} AS {spark_type})"

    # Timestamp
    unit = _arrow_unit(t)
    tz = _arrow_time_zone(t)
    if tz is not None and not _same_time_zone(tz, time_zone):
        raise Skip("arrow_cast to a timestamp in a time zone other than the session's")
    # In a UTC session, Spark's TIMESTAMP holds the same values as an Arrow
    # timestamp without a time zone, and functions such as unix_micros accept
    # only TIMESTAMP, not TIMESTAMP_NTZ.
    utc_session = _is_utc(time_zone)
    spark_type = "TIMESTAMP" if tz is not None or utc_session else "TIMESTAMP_NTZ"
    if kind == "null":
        return f"CAST(NULL AS {spark_type})"
    if kind == "string":
        m = _FRACTION_RE.search(value)
        digits = len(m.group(1).rstrip("0")) if m else 0
        if digits > 6:
            raise Skip("arrow_cast to nanoseconds, which Spark does not have")
        if digits > _UNIT_DIGITS[unit]:
            raise Skip("arrow_cast that truncates a timestamp to seconds or milliseconds")
        return f"CAST({arg_sql} AS {spark_type})"
    # An integer counts units since the epoch.
    if unit == "ns":
        if value % 1000:
            raise Skip("arrow_cast to nanoseconds, which Spark does not have")
        value, unit = value // 1000, "us"
    if tz is None and not utc_session:
        # Spark would print the instant in the session time zone, and Arrow
        # prints a timestamp without a time zone as UTC.
        raise Skip("arrow_cast of a number to a timestamp in a session time zone other than UTC")
    return f"{_EPOCH_FUNCTIONS[unit]}({value})"


# Brackets that nest, so a comma inside `[1, 2]` or `f(a, b)` does not split arguments.
_NESTING = {"(": 1, "[": 1, "{": 1, ")": -1, "]": -1, "}": -1}


# Spark functions that return a DATE.
_DATE_FUNCTIONS = {
    "add_months",
    "current_date",
    "curdate",
    "date_add",
    "date_from_unix_date",
    "date_sub",
    "dateadd",
    "last_day",
    "make_date",
    "next_day",
    "to_date",
    "trunc",
}

_INTEGER_TYPES = ("TINYINT", "SMALLINT", "INT", "BIGINT")


_TIMESTAMP_TYPES = ("TIMESTAMP", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ")


def _visible_type(sql: str) -> Optional[str]:
    """"DATE" or "TIMESTAMP" if a translated expression is visibly one: a typed
    literal, a cast, or a call to a function that returns a DATE. Otherwise None."""
    tokens = _strip_parens(_tokenize(sql))
    if len(tokens) >= 3 and tokens[-2][0] == "cast":
        # expr::TYPE, where `::` applies to the whole of expr
        return _temporal_name(tokens[-1][1]) if _is_operand(tokens[:-2]) else None
    if len(tokens) == 2 and tokens[1][0] == "string":
        return _temporal_name(tokens[0][1])
    # A call: a name followed by its arguments.
    if len(tokens) < 3 or tokens[0][0] != "word" or not _is_operand(tokens):
        return None
    name = tokens[0][1].lower()
    if name == "cast":
        inner = tokens[2:-1]
        if len(inner) >= 2 and inner[-2][1].upper() == "AS":
            return _temporal_name(inner[-1][1])
        return None
    return "DATE" if name in _DATE_FUNCTIONS else None


def _is_operand(tokens: list[tuple[str, str]]) -> bool:
    """Whether tokens are one operand: a single token, or a name followed by a
    parenthesized group that runs to the end, such as `f(a, b)` or `(a + b)`."""
    if len(tokens) == 1:
        return True
    start = 1 if tokens[0][0] == "word" else 0
    if start >= len(tokens) or tokens[start][1] != "(":
        return False
    return _matching_paren(tokens, start) == len(tokens) - 1


def _temporal_name(type_name: str) -> Optional[str]:
    type_name = type_name.upper()
    if type_name == "DATE":
        return "DATE"
    return "TIMESTAMP" if type_name in _TIMESTAMP_TYPES else None


def _matching_paren(tokens: list[tuple[str, str]], open_index: int) -> int:
    depth = 0
    for k in range(open_index, len(tokens)):
        if tokens[k][0] == "other":
            depth += _NESTING.get(tokens[k][1], 0)
            if depth == 0:
                return k
    raise Skip("unbalanced parentheses")


# A Spark type name, such as `timestamp`, `decimal(10, 2)` or `array<int>`.
_SPARK_TYPE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_<>(),:`\s]*")


def _call_args(tokens: list[tuple[str, str]], open_index: int, close: int) -> list:
    """Split the tokens between a call's parentheses on top-level commas."""
    args, start, depth = [], open_index + 1, 0
    for k in range(open_index + 1, close):
        if tokens[k][0] == "other":
            if tokens[k][1] == "," and depth == 0:
                args.append(tokens[start:k])
                start = k + 1
            depth += _NESTING.get(tokens[k][1], 0)
    args.append(tokens[start:close])
    return args


def _type_string(name: str, args: list) -> str:
    """The type argument of arrow_cast or spark_cast, which must be a string literal."""
    if len(args) != 2:
        raise Skip(f"{name} without exactly two arguments")
    type_arg = _strip_parens(args[1])
    if len(type_arg) != 1 or type_arg[0][0] != "string":
        raise Skip(f"{name} to a type that is not a string literal")
    return _unquote(type_arg[0])


_ARROW_TEMPORAL_TYPES = ("Date32", "Date64", "Time32", "Time64", "Timestamp")


def _contains_temporal(t) -> bool:
    if isinstance(t, ArrowType):
        return t.name in _ARROW_TEMPORAL_TYPES or any(_contains_temporal(a) for a in t.args)
    # A struct or map field, or a scalar argument such as a time zone.
    field_type = getattr(t, "type", None)
    return isinstance(field_type, ArrowType) and _contains_temporal(field_type)


# Errors from an Arrow type string with an unexpected shape.
_ARROW_TYPE_ERRORS = (ValueError, TypeError, AttributeError, IndexError)


def _arrow_cast(arg: list, arg_sql: str, type_string: str, time_zone: str) -> str:
    try:
        arrow_type = parse_arrow_type(type_string)
    except _ARROW_TYPE_ERRORS:
        raise Skip("arrow_cast to a type this runner cannot parse")
    if arrow_type.name in _ARROW_TEMPORAL_TYPES:
        return _temporal_cast(arrow_type, arg, arg_sql, time_zone)
    if _contains_temporal(arrow_type) and _literal(arg)[0] != "null":
        # _temporal_cast handles only top-level values. Inside a list or struct,
        # Arrow and Spark convert numbers to dates and timestamps differently.
        raise Skip("arrow_cast to a nested date, time or timestamp")
    try:
        spark_type = spark_type_for_arrow(arrow_type)
    except _ARROW_TYPE_ERRORS:
        raise Skip("arrow_cast to a type this runner cannot parse")
    source = _visible_type(arg_sql) if spark_type in _INTEGER_TYPES else None
    if source == "DATE":
        # Arrow casts a date to an integer as days since the epoch.
        # Spark's CAST returns NULL, and unix_date returns the days.
        days = f"unix_date({arg_sql})"
        return days if spark_type == "INT" else f"CAST({days} AS {spark_type})"
    if source == "TIMESTAMP":
        # Arrow returns the count of the timestamp's unit, and Spark returns seconds.
        raise Skip("arrow_cast of a timestamp to an integer, which counts Arrow's time unit")
    return f"CAST({arg_sql} AS {spark_type})"


def rewrite_casts(sql: str, time_zone: str = "UTC") -> str:
    """Replace each `arrow_cast(expr, 'Type')` and `spark_cast(expr, 'type')`
    with a Spark CAST, or raise Skip.

    `spark_cast` is this crate's emulation of Spark's CAST, and its second
    argument is already a Spark type name.
    """
    tokens = _tokenize(sql)
    out = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        name = text.lower()
        open_index = _next_significant(tokens, i + 1)
        if not (
            kind == "word"
            and name in ("arrow_cast", "spark_cast")
            and open_index < len(tokens)
            and tokens[open_index][1] == "("
        ):
            out.append(text)
            i += 1
            continue
        close = _matching_paren(tokens, open_index)
        args = _call_args(tokens, open_index, close)
        type_string = _type_string(name, args)
        arg_sql = rewrite_casts("".join(t for _, t in args[0]), time_zone).strip()
        if name == "arrow_cast":
            out.append(_arrow_cast(args[0], arg_sql, type_string, time_zone))
        else:
            if not _SPARK_TYPE_RE.fullmatch(type_string.strip()):
                raise Skip("spark_cast to a type that is not a Spark type name")
            out.append(f"CAST({arg_sql} AS {type_string.strip()})")
        i = close + 1
    return "".join(out)


def translate(sql: str, time_zone: str = "UTC") -> str:
    """Translate the parts of DataFusion's SQL that Spark spells differently."""
    tokens = _tokenize(rewrite_casts(sql, time_zone))
    out = []
    # The function name before each open parenthesis, to find CAST(x AS type).
    calls: list[Optional[str]] = []
    last: Optional[tuple[str, str]] = None
    expect_type = False
    for i, (kind, text) in enumerate(tokens):
        if kind == "string" and "\\" in text:
            raise Skip(
                "a string literal contains a backslash, which Spark reads as an "
                "escape and DataFusion does not"
            )
        if kind in ("space", "comment"):
            out.append(text)
            continue
        if kind == "quoted" and text.startswith('"'):
            # DataFusion reads "x" as an identifier, and Spark as a string.
            text = "`" + text[1:-1].replace("`", "``") + "`"
        if kind == "word":
            if expect_type:
                text = _translate_type(tokens, i)
            else:
                m = _COLUMN_RE.fullmatch(text)
                if m:
                    # Spark names VALUES columns col1, col2, ...
                    text = f"col{m.group(1)}"
            expect_type = text.upper() == "AS" and bool(calls) and calls[-1] in (
                "CAST",
                "TRY_CAST",
            )
        elif kind == "cast":
            expect_type = True
        else:
            expect_type = False
            if text == "(":
                calls.append(last[1].upper() if last and last[0] == "word" else None)
            elif text == ")" and calls:
                calls.pop()
        last = (kind, text)
        out.append(text)
    return "".join(out)


_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_CREATE_RE = re.compile(
    rf"\s*CREATE\s+(OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(IF\s+NOT\s+EXISTS\s+)?({_NAME})\s*",
    re.IGNORECASE,
)
_DROP_RE = re.compile(
    rf"\s*DROP\s+(?:TABLE|VIEW)\s+(IF\s+EXISTS\s+)?({_NAME})\s*;?\s*$",
    re.IGNORECASE,
)
_AS_RE = re.compile(r"AS\b\s*", re.IGNORECASE)


def translate_statement(sql: str, time_zone: str = "UTC") -> str:
    """Translate a statement, turning DataFusion tables into Spark temporary views."""
    m = _DROP_RE.match(sql)
    if m:
        return f"DROP VIEW {'IF EXISTS ' if m.group(1) else ''}{m.group(2)}"
    m = _CREATE_RE.match(sql)
    if not m:
        return translate(sql, time_zone)
    if m.group(2):
        # Spark does not allow IF NOT EXISTS on a temporary view.
        raise Skip("CREATE TABLE IF NOT EXISTS, which a Spark temporary view cannot express")
    create = f"CREATE {'OR REPLACE ' if m.group(1) else ''}TEMPORARY VIEW"
    name, rest = m.group(3), sql[m.end() :]
    columns = None
    if rest.startswith("("):
        tokens = _tokenize(rest)
        close = _matching_paren(tokens, 0)
        columns = ["".join(t for _, t in arg).strip() for arg in _call_args(tokens, 0, close)]
        rest = "".join(t for _, t in tokens[close + 1 :]).lstrip()
    m = _AS_RE.match(rest)
    if not m:
        raise Skip("CREATE TABLE without AS")
    query = rest[m.end() :].rstrip().rstrip(";")
    if columns is None:
        return translate(f"{create} {name} AS {query}", time_zone)
    # A newline ends a trailing `--` comment, which would hide the `)`.
    end_comment = "\n" if "--" in query else ""
    names, casts = [], []
    for column in columns:
        parts = column.split(None, 1)
        if len(parts) != 2:
            raise Skip(f"unsupported column definition: {column}")
        names.append(parts[0])
        casts.append(f"CAST({parts[0]} AS {parts[1]}) AS {parts[0]}")
    return translate(
        f"{create} {name} AS SELECT {', '.join(casts)} "
        f"FROM ({query}{end_comment}) AS __slt_values({', '.join(names)})",
        time_zone,
    )


_SET_RE = re.compile(
    r"\s*SET\s+([A-Za-z0-9_.]+)\s*(?:=|\s+TO\s+)\s*(.*?)\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_RESET_RE = re.compile(r"\s*RESET\s+([A-Za-z0-9_.]+)\s*;?\s*$", re.IGNORECASE)


def parse_setting(sql: str) -> Optional[tuple[str, Optional[str]]]:
    """Parse `SET key = value` or `RESET key`. A reset has the value None."""
    m = _SET_RE.match(sql)
    if m:
        value = m.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        return m.group(1).lower(), value
    m = _RESET_RE.match(sql)
    if m:
        return m.group(1).lower(), None
    return None


# ---------------------------------------------------------------------------
# Formatting Spark values the way DataFusion's sqllogictest engine prints them
#
# Top-level values follow datafusion/sqllogictest/src/engines/conversion.rs and
# datafusion_engine/normalize.rs. Values nested in arrays, maps and structs
# follow arrow-rs's ArrayFormatter.
# ---------------------------------------------------------------------------

_SIMPLE_TYPES = {
    "BinaryType": "binary",
    "BooleanType": "boolean",
    "ByteType": "int",
    "CharType": "string",
    "DateType": "date",
    "DoubleType": "double",
    "FloatType": "float",
    "IntegerType": "int",
    "LongType": "int",
    "NullType": "null",
    "ShortType": "int",
    "StringType": "string",
    "TimeType": "time",
    "TimestampNTZType": "timestamp_ntz",
    "TimestampType": "timestamp",
    "VarcharType": "string",
}

# The type letters DataFusion's sqllogictest engine uses.
_TYPE_LETTERS = {
    "boolean": "B",
    "date": "D",
    "decimal": "R",
    "double": "R",
    "float": "R",
    "int": "I",
    "string": "T",
    "time": "D",
    "timestamp": "P",
    "timestamp_ntz": "P",
}


def type_tree(data_type) -> tuple:
    """Describe a PySpark DataType as nested tuples, so formatting needs no PySpark."""
    name = type(data_type).__name__
    if name == "ArrayType":
        return ("array", type_tree(data_type.elementType))
    if name == "MapType":
        return ("map", type_tree(data_type.keyType), type_tree(data_type.valueType))
    if name == "StructType":
        fields = tuple((f.name, type_tree(f.dataType)) for f in data_type.fields)
        return ("struct", fields)
    if name == "DecimalType":
        return ("decimal", data_type.precision, data_type.scale)
    return (_SIMPLE_TYPES.get(name, name),)


def type_letter(tree: tuple) -> str:
    return _TYPE_LETTERS.get(tree[0], "?")


def _contains_timestamp(tree: tuple) -> bool:
    if tree[0] == "timestamp":
        return True
    if tree[0] == "struct":
        return any(_contains_timestamp(t) for _, t in tree[1])
    return any(isinstance(t, tuple) and _contains_timestamp(t) for t in tree[1:])


def _shortest_f32(value: float) -> str:
    """The shortest decimal text that round-trips to the same 32-bit float."""
    for precision in range(1, 10):
        text = f"{value:.{precision}g}"
        try:
            if struct.unpack("<f", struct.pack("<f", float(text)))[0] == value:
                return text
        except OverflowError:
            pass
    return repr(value)


def _round_float(value: float, single: bool, digits: int) -> str:
    """DataFusion's float_to_str: round, drop trailing zeros, no exponent."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    text = _shortest_f32(value) if single else repr(value)
    with localcontext() as ctx:
        ctx.prec = 1000
        # BigDecimal::round rounds half to even.
        rounded = Decimal(text).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_EVEN)
        text = format(rounded.normalize(), "f")
    # BigDecimal has no negative zero.
    return "0" if text == "-0" else text


def _ryu(value: float, single: bool) -> str:
    """Format a float the way the Rust `ryu` crate does (used by arrow-rs)."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0:
        return "-0.0" if math.copysign(1, value) < 0 else "0.0"
    sign, digit_tuple, exponent = Decimal(_shortest_f32(value) if single else repr(value)).as_tuple()
    digits = "".join(map(str, digit_tuple)).rstrip("0")
    exponent += len(digit_tuple) - len(digits)
    length = len(digits)
    kk = length + exponent
    limit, low = (13, -6) if single else (16, -5)
    if 0 <= exponent and kk <= limit:
        text = digits + "0" * exponent + ".0"
    elif 0 < kk <= limit:
        text = digits[:kk] + "." + digits[kk:]
    elif low < kk <= 0:
        text = "0." + "0" * -kk + digits
    elif length == 1:
        text = f"{digits}e{kk - 1}"
    else:
        text = f"{digits[0]}.{digits[1:]}e{kk - 1}"
    return ("-" if sign else "") + text


def _plain_decimal(value: Decimal, scale: int) -> str:
    with localcontext() as ctx:
        ctx.prec = 1000
        return format(value.quantize(Decimal(1).scaleb(-scale)), "f")


def _fraction(microsecond: int) -> str:
    """chrono prints fractional seconds in groups of 3 digits, and none for zero."""
    if microsecond == 0:
        return ""
    if microsecond % 1000 == 0:
        return f".{microsecond // 1000:03d}"
    return f".{microsecond:06d}"


def _naive_timestamp(value: datetime.datetime) -> str:
    return (
        f"{value.year:04d}-{value.month:02d}-{value.day:02d}T"
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
        f"{_fraction(value.microsecond)}"
    )


def zone(name: str) -> datetime.tzinfo:
    """Resolve a Spark/DataFusion time zone name or offset such as +05:30."""
    if _is_utc(name):
        return datetime.timezone.utc
    m = re.fullmatch(r"([+-])(\d{1,2})(?::?(\d{2}))?", name)
    if m:
        offset = datetime.timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0))
        return datetime.timezone(-offset if m.group(1) == "-" else offset)
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise Unsupported(f"time zone {name}")


def _local_timestamp(value: datetime.datetime, time_zone: str) -> str:
    """Spark TIMESTAMP values, which PySpark returns as naive UTC (TZ=UTC)."""
    return _wall_clock(value.replace(tzinfo=datetime.timezone.utc), time_zone)


def _wall_clock(value: datetime.datetime, time_zone: str) -> str:
    """An aware datetime as wall-clock time in time_zone."""
    return _naive_timestamp(value.astimezone(zone(time_zone)).replace(tzinfo=None))


def format_nested(value, tree: tuple, time_zone: str) -> str:
    """Format a value the way arrow-rs's ArrayFormatter displays it."""
    if value is None:
        return "NULL"
    kind = tree[0]
    if kind == "boolean":
        return "true" if value else "false"
    if kind == "int":
        return str(value)
    if kind in ("float", "double"):
        return _ryu(value, kind == "float")
    if kind == "decimal":
        return _plain_decimal(value, tree[2])
    if kind == "string":
        return value
    if kind == "binary":
        return bytes(value).hex()
    if kind == "date":
        return value.isoformat()
    if kind == "timestamp":
        return _local_timestamp(value, time_zone)
    if kind == "timestamp_ntz":
        return _naive_timestamp(value)
    if kind == "time":
        return (
            f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
            f"{_fraction(value.microsecond)}"
        )
    if kind == "array":
        return "[" + ", ".join(format_nested(v, tree[1], time_zone) for v in value) + "]"
    if kind == "map":
        # _maps_as_entries collects each map as a list of (key, value) rows.
        entries = (
            f"{format_nested(k, tree[1], time_zone)}: {format_nested(v, tree[2], time_zone)}"
            for k, v in value
        )
        return "{" + ", ".join(entries) + "}"
    if kind == "struct":
        entries = (
            f"{name}: {format_nested(value[i], t, time_zone)}"
            for i, (name, t) in enumerate(tree[1])
        )
        return "{" + ", ".join(entries) + "}"
    if kind == "null":
        return "NULL"
    raise Unsupported(kind)


def format_cell(value, tree: tuple, time_zone: str) -> str:
    """Format a top-level value the way DataFusion's sqllogictest engine prints it."""
    if value is None:
        return "NULL"
    kind = tree[0]
    if kind == "string":
        if value == "":
            return "(empty)"
        return value.rstrip("\n").replace("\x00", "\\0")
    if kind == "float":
        return _round_float(value, single=True, digits=12)
    if kind == "double":
        # The engine keeps 15 decimal places for the Spark test files.
        return _round_float(value, single=False, digits=15)
    return format_nested(value, tree, time_zone)


def format_rows(rows, trees: list[tuple], time_zone: str) -> list[str]:
    """Format rows as result lines, splitting a multi-line last cell like the engine."""
    lines = []
    for row in rows:
        cells = [format_cell(value, tree, time_zone) for value, tree in zip(row, trees)]
        parts = cells[-1].split("\n") if cells else []
        if len(parts) < 2:
            lines.append(" ".join(cells).rstrip())
            continue
        lines.append(" ".join(cells[:-1]).rstrip())
        for n, part in enumerate(parts, 1):
            content = part.lstrip()
            indent = "-" * (len(part) - len(content))
            lines.append(f"{n:02d}){indent}{content}".rstrip())
    return lines


_ZONED_TIMESTAMP_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})"
)


def localize_timestamps(line: str, time_zone: str) -> str:
    """Rewrite DataFusion timestamps with an offset as wall-clock time in time_zone.

    Spark's TIMESTAMP has no zone of its own, so it is printed in the session
    time zone. DataFusion prints a zoned timestamp in its type's zone with an
    offset (`2024-07-15T00:00:00Z`). This puts both on the same footing.
    """

    def replace(m: re.Match) -> str:
        fraction = m.group(7) or ""
        if len(fraction) > 6 and int(fraction[6:]):
            return m.group(0)
        micro = int(fraction[:6].ljust(6, "0")) if fraction else 0
        offset = datetime.timezone.utc if m.group(8) == "Z" else zone(m.group(8))
        parts = [int(m.group(g)) for g in range(1, 7)]
        return _wall_clock(datetime.datetime(*parts, micro, tzinfo=offset), time_zone)

    return _ZONED_TIMESTAMP_RE.sub(replace, line)


def _localize_expected(lines: list[str], trees: list, time_zone: str) -> list[str]:
    """Localize the timestamps in expected lines, cell by cell when the line
    splits into one cell per column, so that a string column is left alone."""
    is_timestamp = [_contains_timestamp(t) for t in trees]
    result = []
    for line in lines:
        cells = line.split(" ")
        if len(cells) == len(trees):
            cells = [localize_timestamps(c, time_zone) if ts else c for c, ts in zip(cells, is_timestamp)]
            result.append(" ".join(cells))
        else:
            result.append(localize_timestamps(line, time_zone))
    return result


# ---------------------------------------------------------------------------
# Comparing results
# ---------------------------------------------------------------------------


def _sorted(lines: list[str], sort_mode: str) -> list[str]:
    lines = [line.rstrip() for line in lines]
    if sort_mode == "rowsort":
        return sorted(lines)
    if sort_mode == "valuesort":
        return sorted(value for line in lines for value in line.split())
    return lines


def results_match(expected: list[str], actual: list[str], sort_mode: str) -> bool:
    return _sorted(expected, sort_mode) == _sorted(actual, sort_mode)


# A number as DataFusion and this runner print one, without leading zeros,
# underscores or exponents.
_NUMBER_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?|NaN|-?Infinity")


def numeric_difference(expected: list[str], actual: list[str], sort_mode: str) -> Optional[str]:
    """Classify results that differ only in their numbers.

    Returns "scale" if the numbers are equal but printed differently (`7` and
    `7.0`), "close" if some also differ in their last floating-point digits, and
    None if anything else differs.
    """
    expected, actual = _sorted(expected, sort_mode), _sorted(actual, sort_mode)
    if len(expected) != len(actual):
        return None
    kind = "scale"
    for e_line, a_line in zip(expected, actual):
        e_values, a_values = e_line.split(" "), a_line.split(" ")
        if len(e_values) != len(a_values):
            return None
        for e, a in zip(e_values, a_values):
            if e == a:
                continue
            if not (_NUMBER_RE.fullmatch(e) and _NUMBER_RE.fullmatch(a)):
                return None
            e_float, a_float = float(e), float(a)
            if e_float == a_float:
                continue
            if not math.isclose(e_float, a_float, rel_tol=1e-12, abs_tol=1e-15):
                return None
            kind = "close"
    return kind


# ---------------------------------------------------------------------------
# Running records on Spark
# ---------------------------------------------------------------------------


class SparkError(Exception):
    """Spark rejected or failed a query. `kind` is parse, function, table or error."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


class SparkRunner:
    """A local Spark session reset to SPARK_DEFAULTS before each file."""

    def __init__(self):
        # A SPARK_HOME for another Spark version breaks the pip-installed
        # PySpark with "'JavaPackage' object is not callable".
        os.environ.pop("SPARK_HOME", None)
        # PySpark returns TIMESTAMP values in the Python process's time zone.
        os.environ["TZ"] = "UTC"
        if hasattr(time, "tzset"):
            time.tzset()

        install = f"pip install -r {SCRIPT_DIR / 'requirements.txt'}"
        try:
            import pyspark
        except ImportError:
            raise SystemExit(f"error: PySpark is not installed. Install it with `{install}`.")

        if not pyspark.__version__.startswith(SPARK_VERSION + "."):
            raise SystemExit(
                f"error: this runner needs PySpark {SPARK_VERSION}, found "
                f"{pyspark.__version__}. Install it with `{install}`."
            )
        from pyspark.logger import PySparkLogger

        # PySpark logs every failed query to stderr as JSON. The runner reports
        # them itself.
        for name in ("SQLQueryContextLogger", "DataFrameQueryContextLogger"):
            PySparkLogger.getLogger(name).disabled = True
        # Removed when the runner exits.
        self._warehouse = tempfile.TemporaryDirectory(prefix="pyspark-slt-")
        self._start()

    def _start(self) -> None:
        from pyspark.sql import SparkSession

        builder = (
            SparkSession.builder.master("local[1]")
            .appName("datafusion-pyspark-slt-runner")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.sql.warehouse.dir", self._warehouse.name)
            # Spark logs every failed query; the runner reports them instead.
            .config("spark.log.level", "FATAL")
        )
        for key, value in SPARK_DEFAULTS.items():
            builder = builder.config(key, value)
        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel("FATAL")
        self.version = self.spark.version
        self.time_zone = SPARK_DEFAULTS["spark.sql.session.timeZone"]
        # Set when a query runs the JVM out of memory, which leaves it unusable.
        self.broken = False

    def restart(self) -> None:
        """Replace the session with a new JVM, for example after an OutOfMemoryError."""
        from pyspark import SparkContext
        from pyspark.sql import SparkSession

        # A JVM that ran out of memory can hang in a clean stop, so end the process.
        process = getattr(SparkContext._gateway, "proc", None)
        if process is not None:
            process.kill()
            process.wait()
        # PySpark keeps the old gateway and session in class attributes, and
        # would reuse them instead of launching a new JVM.
        SparkContext._gateway = None
        SparkContext._jvm = None
        SparkContext._active_spark_context = None
        SparkSession._instantiatedSession = None
        SparkSession._activeSession = None
        self._start()

    def reset(self) -> None:
        for key, value in SPARK_DEFAULTS.items():
            self.spark.conf.set(key, value)
        self.time_zone = SPARK_DEFAULTS["spark.sql.session.timeZone"]
        for table in self.spark.catalog.listTables():
            if table.isTemporary:
                self.spark.catalog.dropTempView(table.name)

    def set(self, key: str, value: Optional[str]) -> None:
        from pyspark.errors import PySparkException

        if value is None:
            value = SPARK_DEFAULTS[key]
        elif key == "spark.sql.ansi.enabled":
            value = value.lower()
        try:
            self.spark.conf.set(key, value)
        except PySparkException as e:
            raise SparkError("error", _first_line(e))
        if key == "spark.sql.session.timeZone":
            self.time_zone = value

    def run(self, sql: str):
        """Run SQL and return (schema, rows), or raise SparkError."""
        from py4j.protocol import Py4JJavaError
        from pyspark.errors import ParseException, PySparkException

        try:
            df = self.spark.sql(sql)
            schema = df.schema
            # Build the query outside the `try` below, so that a TypeError from
            # the runner's own code is reported as a runner error, not a skip.
            query = _maps_as_entries(df, schema).limit(MAX_ROWS + 1)
            try:
                rows = query.collect()
            except (TypeError, ValueError, OverflowError) as e:
                # For example a MAP with array keys, which a dict cannot hold, or
                # a date outside the years 1 to 9999 that Python supports.
                raise SparkError("convert", f"PySpark cannot convert the result ({e})")
            return schema, rows
        except ParseException as e:
            raise SparkError("parse", f"Spark cannot parse it ({_condition(e)})")
        except PySparkException as e:
            self._check_memory(str(e))
            m = re.search(r"(\w+Type)\.fromInternal is not implemented", str(e))
            if m:
                raise SparkError("convert", f"Spark returned a {m.group(1)} value")
            condition = _condition(e)
            if condition == "UNRESOLVED_ROUTINE":
                name = _parameter(e, "routineName") or "?"
                raise SparkError("function", f"Spark has no function {name}")
            if condition == "TABLE_OR_VIEW_NOT_FOUND":
                raise SparkError("table", "needs a table that Spark could not create")
            raise SparkError("error", _first_line(e))
        except Py4JJavaError as e:
            self._check_memory(str(e.java_exception))
            raise SparkError("error", _first_line(e.java_exception))

    def _check_memory(self, message: str) -> None:
        if "OutOfMemoryError" in message:
            self.broken = True
            raise SparkError("memory", "Spark ran out of memory")


def _maps_as_entries(df, schema):
    """Collect MAP values, at any depth, as arrays of entries.

    PySpark returns a MAP as a dict built from a Java HashMap, which loses
    Spark's entry order and cannot have keys such as arrays.
    """
    if not any(_has_map(f.dataType) for f in schema.fields):
        return df
    from pyspark.sql import functions as F

    names = [f"_c{i}" for i in range(len(schema.fields))]
    columns = [
        _entries(F.col(n), f.dataType).alias(n) for n, f in zip(names, schema.fields)
    ]
    return df.toDF(*names).select(columns)


def _has_map(data_type) -> bool:
    kind = type(data_type).__name__
    if kind == "MapType":
        return True
    if kind == "ArrayType":
        return _has_map(data_type.elementType)
    if kind == "StructType":
        return any(_has_map(f.dataType) for f in data_type.fields)
    return False


def _entries(column, data_type):
    """`column` with each MAP inside it replaced by its array of entries."""
    if not _has_map(data_type):
        return column
    from pyspark.sql import functions as F

    kind = type(data_type).__name__
    if kind == "ArrayType":
        return F.transform(column, lambda x: _entries(x, data_type.elementType))
    if kind == "StructType":
        fields = [
            _entries(column.getField(f.name), f.dataType).alias(f.name)
            for f in data_type.fields
        ]
        return F.when(column.isNotNull(), F.struct(*fields))
    entries = F.map_entries(column)
    if not (_has_map(data_type.keyType) or _has_map(data_type.valueType)):
        return entries
    return F.transform(
        entries,
        lambda e: F.struct(
            _entries(e.getField("key"), data_type.keyType).alias("key"),
            _entries(e.getField("value"), data_type.valueType).alias("value"),
        ),
    )


def _condition(e) -> Optional[str]:
    getter = getattr(e, "getCondition", None) or getattr(e, "getErrorClass", None)
    return getter() if getter else None


def _parameter(e, name: str) -> Optional[str]:
    getter = getattr(e, "getMessageParameters", None)
    parameters = (getter() if getter else None) or {}
    return parameters.get(name)


@dataclass
class Outcome:
    record: Record
    status: str
    detail: str = ""
    spark_sql: Optional[str] = None
    actual: list[str] = field(default_factory=list)


# Statuses that mean DataFusion and Spark disagree, in report order.
DIVERGENT = {
    "mismatch": "results differ",
    "spark-error": "Spark raises an error",
    "no-spark-error": "DataFusion expects an error, Spark returns a result",
    "type": "results match, column types differ",
    "scale": "same numbers, printed with a different scale",
    "close": "results differ only in floating-point digits",
    "order": "same rows in a different order",
    "runner-error": "the runner failed",
}


# Spark's stack() builds all of its rows at once, so stack(2000000000, 1) runs
# the JVM out of memory before the MAX_ROWS limit applies. A row count above
# the INT range is a BIGINT, which Spark rejects before running anything.
_LARGE_STACK_RE = re.compile(r"\bstack\s*\(\s*(\d+)", re.IGNORECASE)
_MAX_INT = 2**31 - 1

# Statements whose output is a plan or schema in each engine's own format.
_ENGINE_SPECIFIC_RE = re.compile(r"\s*(EXPLAIN|DESCRIBE|DESC|SHOW)\b", re.IGNORECASE)


def run_record(runner: SparkRunner, record: Record) -> Outcome:
    if record.skip_reason:
        return Outcome(record, "skip", record.skip_reason)
    setting = parse_setting(record.sql) if record.kind == "statement" else None
    if setting is not None:
        key, value = setting
        if key not in SETTINGS:
            return Outcome(record, "skip", f"no Spark equivalent for {key}")
        try:
            runner.set(SETTINGS[key], value)
        except SparkError as e:
            if record.expect_error:
                return Outcome(record, "pass")
            return Outcome(record, "spark-error", e.message)
        if record.expect_error:
            return Outcome(record, "no-spark-error")
        return Outcome(record, "pass")
    m = _ENGINE_SPECIFIC_RE.match(record.sql)
    if m:
        return Outcome(record, "skip", f"{m.group(1).upper()} output differs between engines")
    try:
        if record.kind == "statement":
            spark_sql = translate_statement(record.sql, runner.time_zone)
        else:
            spark_sql = translate(record.sql, runner.time_zone)
    except Skip as e:
        return Outcome(record, "skip", str(e))
    if any("<slt:ignore>" in line for line in record.expected):
        return Outcome(record, "skip", "uses <slt:ignore>")
    m = _LARGE_STACK_RE.search(spark_sql)
    if m and MAX_ROWS < int(m.group(1)) <= _MAX_INT:
        return Outcome(record, "skip", f"stack() of more than {MAX_ROWS} rows")

    try:
        schema, rows = runner.run(spark_sql)
    except SparkError as e:
        if e.kind != "error":
            return Outcome(record, "skip", e.message, spark_sql)
        if record.expect_error:
            return Outcome(record, "pass", spark_sql=spark_sql)
        return Outcome(record, "spark-error", e.message, spark_sql)

    trees = [type_tree(f.dataType) for f in schema.fields]
    try:
        actual = format_rows(rows[:MAX_ROWS], trees, runner.time_zone)
    except Unsupported as e:
        return Outcome(record, "skip", f"Spark returned a {e} value", spark_sql)
    if len(rows) > MAX_ROWS:
        actual.append(f"... (more than {MAX_ROWS} rows)")
    if record.expect_error:
        return Outcome(record, "no-spark-error", spark_sql=spark_sql, actual=actual)
    if record.kind == "statement":
        return Outcome(record, "pass", spark_sql=spark_sql)

    expected = record.expected
    if not results_match(expected, actual, record.sort_mode) and any(
        _contains_timestamp(t) for t in trees
    ):
        expected = _localize_expected(expected, trees, runner.time_zone)
    if results_match(expected, actual, record.sort_mode):
        letters = "".join(type_letter(t) for t in trees)
        if letters != record.types:
            detail = f"DataFusion `{record.types}`, Spark `{letters}` ({schema.simpleString()})"
            return Outcome(record, "type", detail, spark_sql, actual)
        return Outcome(record, "pass", spark_sql=spark_sql)
    status = numeric_difference(expected, actual, record.sort_mode)
    if status is None:
        status = "order" if results_match(expected, actual, "rowsort") else "mismatch"
    return Outcome(record, status, spark_sql=spark_sql, actual=actual)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _indent(lines, prefix: str = "    ") -> str:
    lines = list(lines) or ["(no rows)"]
    if len(lines) > MAX_REPORT_LINES:
        more = len(lines) - MAX_REPORT_LINES
        lines = lines[:MAX_REPORT_LINES] + [f"... {more} more line(s)"]
    return "\n".join(prefix + line for line in lines)


def report(outcome: Outcome, spark_version: str) -> str:
    record = outcome.record
    location = record.location
    if outcome.status == "skip":
        return f"{location}: skipped: {outcome.detail}"
    if outcome.status == "pass":
        return f"{location}: pass"
    parts = [f"{location}: {DIVERGENT[outcome.status]}", _indent(record.sql.split("\n"))]
    if outcome.spark_sql and outcome.spark_sql != record.sql:
        parts += ["  Spark SQL:", _indent(outcome.spark_sql.split("\n"))]
    if outcome.status in ("mismatch", "scale", "close", "order"):
        parts += ["  DataFusion (expected):", _indent(record.expected)]
        parts += [f"  Spark {spark_version}:", _indent(outcome.actual)]
    elif outcome.status == "no-spark-error":
        parts += [f"  Spark {spark_version}:", _indent(outcome.actual)]
    else:
        parts.append(f"  {outcome.detail}")
    return "\n".join(parts)


def discover(paths: list[str]) -> list[Path]:
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.slt")))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"error: {name} is not a file or directory")
    return files


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Run the datafusion-spark .slt files on Spark {SPARK_VERSION} and "
            "report where Spark disagrees with the expected results."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=[str(DEFAULT_TEST_DIR)],
        help=".slt files or directories (default: all of crates/sqllogictest/test_files)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also list skipped and passing records",
    )
    args = parser.parse_args(argv)
    files = discover(args.paths)
    if not files:
        raise SystemExit("error: no .slt files found")

    # Keep the output complete when it is piped to a file and the run dies.
    sys.stdout.reconfigure(line_buffering=True)
    runner = SparkRunner()
    print(f"Running {len(files)} file(s) on Spark {runner.version}\n")
    statuses, skip_reasons, stopped = run_files(runner, files, args.verbose)

    total = sum(statuses.values())
    print(f"Spark {runner.version}: {total} records in {len(files)} file(s)")
    print(f"  {statuses['pass']:6d}  pass")
    for status, description in DIVERGENT.items():
        if statuses[status]:
            print(f"  {statuses[status]:6d}  {description}")
    print(f"  {statuses['skip']:6d}  skipped")
    for reason, count in skip_reasons.most_common(MAX_SKIP_REASONS):
        print(f"  {count:14d}  {reason}")
    if len(skip_reasons) > MAX_SKIP_REASONS:
        more = len(skip_reasons) - MAX_SKIP_REASONS
        print(f"  {'':>14}  ... {more} more reasons (use --verbose)")
    if stopped:
        print(f"\nStopped early: {stopped}")
        return 2
    return 1 if any(statuses[s] for s in DIVERGENT) else 0


# A run gives up after the Spark session dies this many times.
MAX_RESTARTS = 5


def run_files(runner, files: list[Path], verbose: bool = False):
    """Run every record in `files`, printing each difference.

    Returns (statuses, skip_reasons, stopped). `stopped` says why the run ended
    early, or is None.
    """
    statuses: Counter = Counter()
    skip_reasons: Counter = Counter()
    restarts = 0

    def new_session(path: Path, reason: str) -> Optional[str]:
        """Replace a session that failed, returning why the run must stop, if it must."""
        nonlocal restarts
        if restarts == MAX_RESTARTS:
            return f"{reason}, and the Spark session has already been replaced {MAX_RESTARTS} times"
        restarts += 1
        print(f"{reason}. Starting a new Spark session before {_display_path(path)}\n")
        try:
            runner.restart()
            runner.reset()
        except Exception as e:
            return f"a new Spark session failed to start: {_first_line(e)}"
        return None

    broken = False
    for path in files:
        stopped = None
        if broken:
            stopped = new_session(path, "Spark failed in the previous file")
        else:
            try:
                runner.reset()
            except Exception as e:
                stopped = new_session(path, f"Spark failed ({_first_line(e)})")
        if stopped:
            return statuses, skip_reasons, stopped
        broken = False
        for record in parse_slt(path):
            if broken:
                # A new session would not have this file's views and settings.
                outcome = Outcome(record, "skip", "Spark failed earlier in the file")
            else:
                try:
                    outcome = run_record(runner, record)
                except Exception as e:
                    if _session_died(e):
                        broken = True
                        print(f"{record.location}: the Spark session died, skipping the rest of the file\n")
                        outcome = Outcome(record, "skip", "the Spark session died while running it")
                    else:
                        # A bug in this runner must never look like Spark behavior.
                        detail = traceback.format_exc(limit=1).strip().split("\n")[-1]
                        outcome = Outcome(record, "runner-error", detail)
                if not broken and getattr(runner, "broken", False):
                    broken = True
                    print(f"{record.location}: Spark ran out of memory, skipping the rest of the file\n")
            statuses[outcome.status] += 1
            if outcome.status == "skip":
                skip_reasons[outcome.detail] += 1
            if outcome.status in DIVERGENT:
                print(report(outcome, runner.version) + "\n")
            elif verbose:
                print(report(outcome, runner.version))
    return statuses, skip_reasons, None


def _first_line(e: BaseException) -> str:
    try:
        return str(e).strip().split("\n")[0]
    except Exception:
        return type(e).__name__


def _session_died(e: BaseException) -> bool:
    """True if the error means the connection to the Spark JVM is gone."""
    chain = [c for c in (e, e.__cause__, e.__context__) if c is not None]
    names = {type(c).__name__ for c in chain}
    if names & {"Py4JNetworkError", "ConnectionRefusedError", "ConnectionResetError", "BrokenPipeError"}:
        return True
    messages = []
    for c in chain:
        try:
            messages.append(str(c))
        except Exception:
            pass
    text = " ".join(messages)
    return "Answer from Java side is empty" in text or "Error while sending or receiving" in text


if __name__ == "__main__":
    sys.exit(main())
