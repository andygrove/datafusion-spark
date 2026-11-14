#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Copies an approved release candidate from the "dev" area of the dist.apache.org
# datafusion repository to the "release" area.
#
# Only run this after the Apache DataFusion PMC has approved the release.
#
# Adapted from https://github.com/apache/datafusion-sqlparser-rs/blob/main/dev/release/release-tarball.sh
#
# See dev/release/README.md for the full release instructions.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <version> <rc>"
  echo "ex. $0 56.0.0 1"
  exit 1
fi

version=$1
rc=$2

read -r -p "Proceed to release tarball for ${version}-rc${rc}? [y/N]: " answer
answer=${answer:-no}
if [ "${answer}" != "y" ]; then
  echo "Cancelled tarball release!"
  exit 1
fi

tmp_dir=$(mktemp -d -t apache-datafusion-spark-dist.XXXXX)
trap 'rm -rf "${tmp_dir}"' EXIT

echo "Checkout dev dist repository"
svn co \
  "https://dist.apache.org/repos/dist/dev/datafusion/apache-datafusion-spark-${version}-rc${rc}" \
  "${tmp_dir}/dev"

echo "Checkout release dist repository"
svn co --depth=immediates https://dist.apache.org/repos/dist/release/datafusion "${tmp_dir}/release"

release_version=datafusion-spark-${version}
if [ -e "${tmp_dir}/release/${release_version}" ]; then
  echo "Cannot continue: ${release_version} already exists in the release area" >&2
  exit 1
fi

echo "Copy ${version}-rc${rc} to release working copy"
mkdir -p "${tmp_dir}/release/${release_version}"
cp "${tmp_dir}"/dev/apache-datafusion-spark-* "${tmp_dir}/release/${release_version}/"
svn add "${tmp_dir}/release/${release_version}"

echo "Commit release"
svn ci -m "Apache DataFusion Spark ${version}" "${tmp_dir}/release"

echo "Success! The release is available here:"
echo "  https://dist.apache.org/repos/dist/release/datafusion/${release_version}"
