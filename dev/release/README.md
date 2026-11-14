<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Release Process

This document describes how to make a release of Apache DataFusion Spark (the `datafusion-spark` crate).

## Process Overview

Under the Apache governance model, an official release is a signed source tarball that the
DataFusion PMC has approved. The `datafusion-spark` crate is published to crates.io from the
approved tarball after the vote passes.

The steps are:

1. Prepare a release PR that sets the version, pins the dependencies to released crates, and adds
   the changelog.
2. Tag the merged commit as a release candidate (RC), then create, sign, and upload the tarball.
3. Hold a vote on the dev@datafusion.apache.org mailing list.
4. After the vote passes, move the tarball to the release area, tag the release, and publish to
   crates.io.

The scripts in this directory are:

| Script                        | Purpose                                                                 |
| ----------------------------- | ----------------------------------------------------------------------- |
| `generate-changelog.py`       | Generates the changelog from GitHub PRs merged between two git refs     |
| `create-tarball.sh`           | Creates, signs, and uploads the RC tarball, and prints the vote email   |
| `verify-release-candidate.sh` | Verifies the signature and checksums of an RC, then builds and tests it |
| `release-tarball.sh`          | Moves an approved RC from the dev area to the release area in svn      |
| `run-rat.sh`                  | Runs the Apache RAT license check on the repository or on a tarball     |

## Versioning

The version of `datafusion-spark` matches the major version of the DataFusion release it is built
against. For example, `datafusion-spark` 56.x depends on DataFusion 56.x. The version is set once
for the whole workspace in the root `Cargo.toml` (`[workspace.package] version`).

crates.io does not accept crates with git dependencies. During development `main` may depend on a
DataFusion git branch (for example `branch = "branch-56"`); before a release these must be changed to
the matching DataFusion release published on crates.io.

## Prerequisites

Only committers can upload release candidates, and only PMC members can finalize a release.

You need:

- A GPG key for signing, with the public key added to the
  [DataFusion KEYS file](https://dist.apache.org/repos/dist/release/datafusion/KEYS). If it is not
  there yet, follow the instructions at the top of that file and commit it with svn (PMC members can
  commit to the release area, so ask one if you cannot).
- `svn`, with your Apache credentials.
- Java, for the Apache RAT license check.
- Python 3 with `PyGitHub` (`pip3 install PyGitHub`), for the changelog.
- A [GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
  for the changelog script. It only needs read access to public repositories.

## Prepare the Release PR

Start from the latest `main` in the `apache` remote:

```shell
git fetch apache
git checkout -b release-56.0.0 apache/main
```

### Update the version and dependencies

1. Set `version` under `[workspace.package]` in the root `Cargo.toml` to the release version (for
   example `56.0.0`).
2. Replace any DataFusion git dependencies in `[workspace.dependencies]` with the released versions
   from crates.io (for example `datafusion = { version = "56.0.0", default-features = false }`), and
   make sure the `arrow` version matches the one DataFusion uses.
3. Run `cargo update -p datafusion` (and the other changed crates) so that `Cargo.lock` is updated,
   then run the checks from `AGENTS.md`:

   ```shell
   cargo fmt --all
   cargo clippy --all-targets --all-features -- -D warnings
   cargo test -p datafusion-spark
   cargo test -p datafusion-spark-sqllogictest
   cargo publish --dry-run -p datafusion-spark
   ```

### Generate the changelog

The changelog lists the PRs merged since the previous release. PRs are categorized by their GitHub
labels and their [Conventional Commits](https://www.conventionalcommits.org/) title prefix (`feat:`,
`fix:`, `docs:`, `perf:`, and `!` for breaking changes), so it is worth fixing up mislabelled PR
titles before generating it.

Pass the previous release tag, the ref to release, and the new version:

```shell
export GITHUB_TOKEN=<your-token>
mkdir -p dev/changelog
python3 dev/release/generate-changelog.py v55.0.0 HEAD 56.0.0 > dev/changelog/56.0.0.md
```

For the first release, use the first commit of the repository as the previous ref
(`git rev-list --max-parents=0 HEAD`).

Review the generated file, then commit everything and open a PR against `main`:

```shell
git commit -a -m "chore: prepare 56.0.0 release"
git push <your-fork> release-56.0.0
```

## Create the Release Candidate

After the release PR is merged, create the release candidate artifacts from the merged commit.

### Pick a release candidate number

RC numbers start at `1` for each version and increase by one for every new candidate.

### Tag the release candidate

The official release artifact is the signed tarball, but we also tag the commit for convenience
and code archaeology:

```shell
git fetch apache
git tag v56.0.0-rc1 apache/main
git push apache v56.0.0-rc1
```

### Create, sign, and upload the tarball

```shell
./dev/release/create-tarball.sh 56.0.0 1
```

The script:

1. Checks that the tag exists and that its `Cargo.toml` has the release version.
2. Creates `dev/dist/apache-datafusion-spark-56.0.0-rc1/apache-datafusion-spark-56.0.0.tar.gz`
   from the tagged commit with `git archive`.
3. Runs the Apache RAT license check on the tarball.
4. Signs the tarball with GPG and writes `.sha256` and `.sha512` checksums.
5. Uploads the files to
   [the DataFusion dev dist area](https://dist.apache.org/repos/dist/dev/datafusion).
6. Prints the vote email to send to the dev list.

`dev/dist` is ignored by git and can be removed after the upload.

## Vote on the Release Candidate

Send the email printed by `create-tarball.sh` to dev@datafusion.apache.org. The vote stays open for
at least 72 hours, and the release needs at least three binding +1 votes from PMC members and more
binding +1 than -1 votes.

When the vote ends, reply with a `[RESULT] [VOTE] Release Apache DataFusion Spark 56.0.0 RC1` email
that tallies the votes.

### Verifying release candidates

Anyone can help verify a release candidate. The `verify-release-candidate.sh` script downloads the
RC, checks its signature and checksums, installs a Rust toolchain in a temporary directory, and then
checks formatting, runs the tests, and runs `cargo publish --dry-run`:

```shell
./dev/release/verify-release-candidate.sh 56.0.0 1
```

Set `VERIFY_TMPDIR` to keep the working directory after the script finishes.

Beyond the script, check that the `LICENSE.txt` and `NOTICE.txt` files are present and correct and
that the tarball does not contain any compiled code.

### If the release is not approved

Fix the problems found during the vote, merge the fixes into `main`, and start again from
[Create the Release Candidate](#create-the-release-candidate) with the next RC number.

## Finalize the Release

These steps can only be done by PMC members.

### Move the tarball to the release area

```shell
./dev/release/release-tarball.sh 56.0.0 1
```

This copies the approved RC to
`https://dist.apache.org/repos/dist/release/datafusion/datafusion-spark-56.0.0`.

### Tag the release

Point the release tag at the approved RC tag:

```shell
git tag v56.0.0 v56.0.0-rc1
git push apache v56.0.0
```

The release is now official.

### Publish to crates.io

Only approved release tarballs may be published to crates.io, to comply with Apache Software
Foundation governance.

If you have not published the crate before, follow
[the cargo publishing instructions](https://doc.rust-lang.org/cargo/reference/publishing.html) to
create an account and log in to crates.io, then ask an existing owner to add you to the
[`datafusion-spark` crate](https://crates.io/crates/datafusion-spark).

Download and unpack the official release tarball, check that the root `Cargo.toml` has the release
version, and publish from inside it:

```shell
curl -O https://dist.apache.org/repos/dist/release/datafusion/datafusion-spark-56.0.0/apache-datafusion-spark-56.0.0.tar.gz
tar xzf apache-datafusion-spark-56.0.0.tar.gz
cd apache-datafusion-spark-56.0.0
cargo publish -p datafusion-spark
```

`datafusion-spark-sqllogictest` has `publish = false` and is not published.

### Clean up

Remove the release candidates from the dev area:

```shell
svn delete -m "Remove Apache DataFusion Spark 56.0.0 release candidates" \
  https://dist.apache.org/repos/dist/dev/datafusion/apache-datafusion-spark-56.0.0-rc1
```

Only the latest release of each project should stay in the release area. Older releases are
archived automatically at https://archive.apache.org/dist/datafusion/, so remove the previous one:

```shell
svn delete -m "Remove Apache DataFusion Spark 55.0.0" \
  https://dist.apache.org/repos/dist/release/datafusion/datafusion-spark-55.0.0
```

### Record the release and announce it

1. Add the release at https://reporter.apache.org/addrelease.html?datafusion so it appears in the
   PMC board reports.
2. Create a [GitHub release](https://github.com/apache/datafusion-spark/releases) for the `v56.0.0`
   tag, linking to the changelog.
3. Send an `[ANNOUNCE] Apache DataFusion Spark 56.0.0 released` email to dev@datafusion.apache.org
   and announce@apache.org (from your apache.org address).
