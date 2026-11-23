#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Regenerates docs/functions from the #[user_doc] attributes on each function.
#
# Usage: dev/update_function_docs.sh

set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

TARGET_DIR="docs/functions"
mkdir -p "$(dirname "${TARGET_DIR}")"

# Generate into a temporary directory so a failed build leaves the existing reference untouched,
# and so pages for removed sections don't linger.
TMP_DIR="$(mktemp -d "${TARGET_DIR}.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

cargo run --quiet -p datafusion-spark-docgen --bin print_functions_docs -- "${TMP_DIR}"

chmod 755 "${TMP_DIR}"
rm -rf "${TARGET_DIR}"
mv "${TMP_DIR}" "${TARGET_DIR}"

echo "Wrote ${TARGET_DIR}"
