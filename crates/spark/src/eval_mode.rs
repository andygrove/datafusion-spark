// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::config::ConfigOptions;

/// How a Spark expression handles a value it cannot evaluate, such as an
/// overflow or a string that does not parse.
///
/// Mirrors Spark's
/// [`EvalMode`](https://github.com/apache/spark/blob/v4.1.1/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/EvalMode.scala).
/// An expression that supports more than one mode takes an `EvalMode` rather
/// than a `bool`, so that a `try_*` function can share its implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EvalMode {
    /// Spark's behavior when `spark.sql.ansi.enabled` is `false`, the default
    /// before Spark 4.0. An invalid value usually produces `NULL` or a wrapped
    /// result instead of an error.
    #[default]
    Legacy,
    /// Spark's behavior when `spark.sql.ansi.enabled` is `true`, the default
    /// since Spark 4.0. An invalid value raises an error.
    Ansi,
    /// The mode of Spark's `try_*` functions, such as `try_cast` and
    /// `try_parse_url`. It is the same as [`EvalMode::Ansi`], except that an
    /// error on a value produces `NULL` instead.
    Try,
}

impl EvalMode {
    /// Returns [`EvalMode::Ansi`] when `ansi_enabled` is true, and
    /// [`EvalMode::Legacy`] otherwise, like Spark's `EvalMode.fromBoolean`.
    pub fn from_ansi_enabled(ansi_enabled: bool) -> Self {
        if ansi_enabled {
            Self::Ansi
        } else {
            Self::Legacy
        }
    }

    /// Returns the mode for a session, from
    /// `datafusion.execution.enable_ansi_mode`, like Spark's
    /// `EvalMode.fromSQLConf`. A session never selects [`EvalMode::Try`]; only a
    /// `try_*` function does.
    pub fn from_config(config: &ConfigOptions) -> Self {
        Self::from_ansi_enabled(config.execution.enable_ansi_mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_legacy() {
        assert_eq!(EvalMode::default(), EvalMode::Legacy);
    }

    #[test]
    fn test_from_ansi_enabled() {
        assert_eq!(EvalMode::from_ansi_enabled(false), EvalMode::Legacy);
        assert_eq!(EvalMode::from_ansi_enabled(true), EvalMode::Ansi);
    }

    #[test]
    fn test_from_config() {
        let mut config = ConfigOptions::default();
        assert_eq!(EvalMode::from_config(&config), EvalMode::Legacy);

        config.execution.enable_ansi_mode = true;
        assert_eq!(EvalMode::from_config(&config), EvalMode::Ansi);
    }
}
