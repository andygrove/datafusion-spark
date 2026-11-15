#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

# Creates a signed source tarball in
# dev/dist/apache-datafusion-spark-<version>-rc<rc>/apache-datafusion-spark-<version>.tar.gz,
# uploads it to the "dev" area of the dist.apache.org datafusion repository, and prints
# an email to send to the dev@datafusion.apache.org list for a formal vote.
#
# Adapted from https://github.com/apache/datafusion-sqlparser-rs/blob/main/dev/release/create-tarball.sh
#
# See dev/release/README.md for the full release instructions.
#
# Requirements:
#
# 1. gpg set up for signing, with your public key added to
#    https://dist.apache.org/repos/dist/release/datafusion/KEYS
# 2. svn installed and credentials for the Apache svn server
# 3. Java (for the Apache RAT license check)

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_TOP_DIR="$(cd "${SOURCE_DIR}/../../" && pwd)"

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <version> <rc>"
  echo "ex. $0 56.0.0 1"
  exit 1
fi

version=$1
rc=$2
tag="v${version}-rc${rc}"

release=apache-datafusion-spark-${version}
distdir=${SOURCE_TOP_DIR}/dev/dist/${release}-rc${rc}
tarname=${release}.tar.gz
tarball=${distdir}/${tarname}
url="https://dist.apache.org/repos/dist/dev/datafusion/${release}-rc${rc}"

echo "Attempting to create ${tarball} from tag ${tag}"
if ! release_hash=$(cd "${SOURCE_TOP_DIR}" && git rev-list --max-count=1 "${tag}" 2>/dev/null); then
  echo "Cannot continue: unknown git tag: ${tag}" >&2
  exit 1
fi

# The workspace version in the tagged Cargo.toml must match the release version
tagged_version=$(cd "${SOURCE_TOP_DIR}" && git show "${release_hash}:Cargo.toml" \
  | sed -n 's/^version = "\(.*\)"/\1/p' | head -n 1)
if [ "${tagged_version}" != "${version}" ]; then
  echo "Cannot continue: Cargo.toml at ${tag} has version '${tagged_version}', expected '${version}'" >&2
  exit 1
fi

if [ -d "${distdir}" ]; then
  echo "Cannot continue: ${distdir} already exists" >&2
  exit 1
fi

# Create the tarball containing the files in git at ${release_hash}.
# The files in the tarball are prefixed with ${release}/
mkdir -p "${distdir}"
(cd "${SOURCE_TOP_DIR}" && git archive "${release_hash}" --prefix "${release}/" | gzip > "${tarball}")

echo "Running RAT license checker on ${tarball}"
"${SOURCE_DIR}/run-rat.sh" "${tarball}"

echo "Signing tarball and creating checksums"
gpg --armor --output "${tarball}.asc" --detach-sig "${tarball}"
# Use the relative path of the tarball so that the checksums can be verified with
#   shasum -a 512 --check apache-datafusion-spark-<version>.tar.gz.sha512
(cd "${distdir}" && shasum -a 256 "${tarname}") > "${tarball}.sha256"
(cd "${distdir}" && shasum -a 512 "${tarname}") > "${tarball}.sha512"

echo "Uploading to ${url}"
svn co --depth=empty https://dist.apache.org/repos/dist/dev/datafusion "${SOURCE_TOP_DIR}/dev/dist"
svn add "${distdir}"
svn ci -m "Apache DataFusion Spark ${version} RC${rc}" "${distdir}"

echo ""
echo "Draft email for dev@datafusion.apache.org mailing list"
echo ""
echo "---------------------------------------------------------"
cat <<MAIL
To: dev@datafusion.apache.org
Subject: [VOTE] Release Apache DataFusion Spark ${version} RC${rc}

Hi,

I would like to propose a release of Apache DataFusion Spark version ${version}.

This release candidate is based on commit: ${release_hash} [1]
The proposed release tarball and signatures are hosted at [2].
The changelog is located at [3].

Please download, verify checksums and signatures, run the unit tests, and vote
on the release. The vote will be open for at least 72 hours.

Only votes from PMC members are binding, but all members of the community are
encouraged to test the release and vote with "(non-binding)".

The standard verification procedure is documented at
https://github.com/apache/datafusion-spark/blob/main/dev/release/README.md#verifying-release-candidates

[ ] +1 Release this as Apache DataFusion Spark ${version}
[ ] +0
[ ] -1 Do not release this as Apache DataFusion Spark ${version} because...

Here is my vote:

+1

[1]: https://github.com/apache/datafusion-spark/tree/${release_hash}
[2]: ${url}
[3]: https://github.com/apache/datafusion-spark/blob/${release_hash}/dev/changelog/${version}.md
MAIL
echo "---------------------------------------------------------"
