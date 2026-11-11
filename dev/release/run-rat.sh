#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Runs Apache RAT (Release Audit Tool) over the git-tracked files to verify license headers.

set -euo pipefail

RAT_VERSION=0.17
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

RAT_JAR="${WORK_DIR}/apache-rat-${RAT_VERSION}.jar"
curl -fsSL -o "${RAT_JAR}" \
  "https://repo1.maven.org/maven2/org/apache/rat/apache-rat/${RAT_VERSION}/apache-rat-${RAT_VERSION}.jar"

# Only audit tracked files, so build output and local files are ignored
mkdir "${WORK_DIR}/src"
git -C "${ROOT_DIR}" archive HEAD | tar -x -C "${WORK_DIR}/src"

java -jar "${RAT_JAR}" -d "${WORK_DIR}/src" -E "${ROOT_DIR}/dev/release/rat_exclude_files.txt" \
  | tee "${WORK_DIR}/rat.txt"

if grep -qE '^ *[1-9][0-9]* Unknown Licenses|^Unapproved licenses:|^ *Unapproved: *[1-9]' "${WORK_DIR}/rat.txt"; then
  echo "RAT found files without approved license headers" >&2
  exit 1
fi
