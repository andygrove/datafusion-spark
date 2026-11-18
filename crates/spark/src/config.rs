// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::str::FromStr;

use datafusion_common::config::{ConfigExtension, ConfigField, ConfigOptions, Visit};
use datafusion_common::extensions_options;
use datafusion_common::{DataFusionError, Result};

/// Policy for handling duplicate keys in Spark map functions.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum MapKeyDedupPolicy {
    #[default]
    Exception,
    LastWin,
}

impl FromStr for MapKeyDedupPolicy {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "EXCEPTION" => Ok(Self::Exception),
            "LAST_WIN" => Ok(Self::LastWin),
            other => Err(DataFusionError::Configuration(format!(
                "Invalid MapKeyDedupPolicy: {other}. Expected one of: EXCEPTION, LAST_WIN"
            ))),
        }
    }
}

impl ConfigField for MapKeyDedupPolicy {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        v.some(key, self, description)
    }

    fn set(&mut self, _: &str, value: &str) -> Result<()> {
        *self = value.parse()?;
        Ok(())
    }
}

impl fmt::Display for MapKeyDedupPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exception => write!(f, "EXCEPTION"),
            Self::LastWin => write!(f, "LAST_WIN"),
        }
    }
}

extensions_options! {
    /// Spark-specific configuration options under the `spark` namespace.
    pub struct SparkOptions {
        /// How to handle duplicate map keys: EXCEPTION raises an error (default),
        /// while LAST_WIN keeps the last value for each key.
        pub map_key_dedup_policy: MapKeyDedupPolicy, default = MapKeyDedupPolicy::Exception
    }
}

impl ConfigExtension for SparkOptions {
    const PREFIX: &'static str = "spark";
}

/// Returns whether Spark map functions should keep the last value for duplicate keys.
///
/// The new key is `spark.map_key_dedup_policy`. For DataFusion 56 compatibility,
/// the legacy `datafusion.spark.map_key_dedup_policy` is also honored: LAST_WIN
/// on either key takes precedence over EXCEPTION on the other.
/// Remove the legacy lookup when upgrading to DataFusion 57.
pub fn map_last_value_wins(options: &ConfigOptions) -> bool {
    options
        .extensions
        .get::<SparkOptions>()
        .is_some_and(|options| options.map_key_dedup_policy == MapKeyDedupPolicy::LastWin)
        || options.spark.map_key_dedup_policy
            == datafusion_common::config::MapKeyDedupPolicy::LastWin
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::config::ConfigOptions;

    #[test]
    fn spark_options_use_spark_prefix() {
        let mut config = ConfigOptions::default();
        config.extensions.insert(SparkOptions::default());

        config
            .set("spark.map_key_dedup_policy", "last_win")
            .unwrap();

        assert_eq!(
            config
                .extensions
                .get::<SparkOptions>()
                .unwrap()
                .map_key_dedup_policy,
            MapKeyDedupPolicy::LastWin
        );
    }

    #[test]
    fn spark_options_reject_invalid_policy() {
        let mut config = ConfigOptions::default();
        config.extensions.insert(SparkOptions::default());

        let error = config
            .set("spark.map_key_dedup_policy", "bogus")
            .unwrap_err()
            .to_string();

        assert!(error.contains("Invalid MapKeyDedupPolicy"), "{error}");
    }

    #[test]
    fn missing_spark_options_use_exception_policy() {
        assert!(!map_last_value_wins(&ConfigOptions::default()));
    }

    #[test]
    fn configured_spark_options_enable_last_win() {
        let mut config = ConfigOptions::default();
        config.extensions.insert(SparkOptions {
            map_key_dedup_policy: MapKeyDedupPolicy::LastWin,
        });

        assert!(map_last_value_wins(&config));
    }

    #[test]
    fn legacy_and_extension_policy_combinations() {
        for legacy in ["EXCEPTION", "LAST_WIN"] {
            for extension in [None, Some("EXCEPTION"), Some("LAST_WIN")] {
                let mut config = ConfigOptions::default();
                config
                    .set("datafusion.spark.map_key_dedup_policy", legacy)
                    .unwrap();
                if let Some(policy) = extension {
                    config.extensions.insert(SparkOptions::default());
                    config.set("spark.map_key_dedup_policy", policy).unwrap();
                }

                assert_eq!(
                    map_last_value_wins(&config),
                    legacy == "LAST_WIN" || extension == Some("LAST_WIN"),
                    "legacy={legacy}, extension={extension:?}"
                );
            }
        }
    }

    #[test]
    fn map_policy_has_description() {
        let entries = datafusion_common::config::ExtensionOptions::entries(
            &SparkOptions::default(),
        );
        let entry = entries
            .iter()
            .find(|entry| entry.key == "map_key_dedup_policy")
            .unwrap();
        assert!(entry.description.contains("EXCEPTION"));
        assert!(entry.description.contains("LAST_WIN"));
    }
}
