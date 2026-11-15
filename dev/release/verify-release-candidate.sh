#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Verifies a release candidate: downloads the tarball, checks its signature and
# checksums, then builds and tests the source in a fresh Rust toolchain.
#
# Set VERIFY_TMPDIR to keep the working directory after the script finishes.
#
# Adapted from https://github.com/apache/datafusion-sqlparser-rs/blob/main/dev/release/verify-release-candidate.sh
#
# See dev/release/README.md for the full release instructions.

case $# in
  2) VERSION="$1"
     RC_NUMBER="$2"
     ;;
  *) echo "Usage: $0 X.Y.Z RC_NUMBER"
     echo "ex. $0 56.0.0 1"
     exit 1
     ;;
esac

set -euxo pipefail

DIST_URL='https://dist.apache.org/repos/dist'
DIST_NAME="apache-datafusion-spark-${VERSION}"

download_dist_file() {
  curl --silent --show-error --fail --location --remote-name "$1"
}

download_rc_file() {
  download_dist_file "${DIST_URL}/dev/datafusion/${DIST_NAME}-rc${RC_NUMBER}/$1"
}

import_gpg_keys() {
  download_dist_file "${DIST_URL}/release/datafusion/KEYS"
  gpg --import KEYS
}

if type shasum >/dev/null 2>&1; then
  sha256_verify="shasum -a 256 -c"
  sha512_verify="shasum -a 512 -c"
else
  sha256_verify="sha256sum -c"
  sha512_verify="sha512sum -c"
fi

fetch_archive() {
  download_rc_file "${DIST_NAME}.tar.gz"
  download_rc_file "${DIST_NAME}.tar.gz.asc"
  download_rc_file "${DIST_NAME}.tar.gz.sha256"
  download_rc_file "${DIST_NAME}.tar.gz.sha512"

  gpg --verify "${DIST_NAME}.tar.gz.asc" "${DIST_NAME}.tar.gz"
  ${sha256_verify} "${DIST_NAME}.tar.gz.sha256"
  ${sha512_verify} "${DIST_NAME}.tar.gz.sha512"
}

setup_tempdir() {
  # shellcheck disable=SC2329 # invoked via trap
  cleanup() {
    if [ "${TEST_SUCCESS}" = "yes" ]; then
      rm -rf "${VERIFY_TMPDIR}"
    else
      echo "Failed to verify release candidate. See ${VERIFY_TMPDIR} for details."
    fi
  }

  if [ -z "${VERIFY_TMPDIR:-}" ]; then
    # clean up automatically if VERIFY_TMPDIR is not defined
    VERIFY_TMPDIR=$(mktemp -d -t "$1.XXXXX")
    trap cleanup EXIT
  else
    # don't clean up automatically
    mkdir -p "${VERIFY_TMPDIR}"
  fi
}

test_source_distribution() {
  # Install a Rust toolchain local to the sandbox; rust-toolchain / rust-version
  # in the source determine what is needed
  export RUSTUP_HOME=$PWD/../test-rustup
  export CARGO_HOME=$PWD/../test-rustup

  curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --profile minimal \
    --component rustfmt

  # shellcheck disable=SC1091
  source "${CARGO_HOME}/env"

  # Raises on any formatting errors
  cargo fmt --all -- --check

  cargo build --all-targets --all-features
  cargo test -p datafusion-spark --all-features
  cargo test -p datafusion-spark-sqllogictest

  # Released crates must only depend on crates published to crates.io
  if grep -n 'git = ' Cargo.toml crates/*/Cargo.toml; then
    echo "Cargo.toml must not contain git dependencies for releases"
    exit 1
  fi

  # Check that the crate can be packaged for crates.io
  cargo publish --dry-run -p datafusion-spark
}

TEST_SUCCESS=no

setup_tempdir "${DIST_NAME}"
echo "Working in sandbox ${VERIFY_TMPDIR}"
cd "${VERIFY_TMPDIR}"

import_gpg_keys
fetch_archive
tar xf "${DIST_NAME}.tar.gz"
pushd "${DIST_NAME}"
  test_source_distribution
popd

TEST_SUCCESS=yes
echo 'Release candidate looks good!'
exit 0
