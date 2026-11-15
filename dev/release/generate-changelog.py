#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Generate a changelog from the GitHub pull requests merged between two git refs.

Adapted from https://github.com/apache/datafusion-sqlparser-rs/blob/main/dev/release/generate-changelog.py

Usage:
    export GITHUB_TOKEN=<token>
    python3 dev/release/generate-changelog.py <previous-tag> <current-ref> <version> > dev/changelog/<version>.md
"""

import argparse
import os
import re
import subprocess
import sys

from github import Auth, Github

PROJECT = "apache/datafusion-spark"


def print_pulls(repo_name, title, pulls):
    if len(pulls) > 0:
        print(f"**{title}:**")
        print()
        for pull, commit in pulls:
            url = f"https://github.com/{repo_name}/pull/{pull.number}"
            author = f"({commit.author.login})" if commit.author else ""
            print(f"- {pull.title} [#{pull.number}]({url}) {author}")
        print()


def git(*args):
    return subprocess.check_output(["git", *args], text=True)


def generate_changelog(repo, repo_name, tag1, tag2, version):
    # get a list of commits between two refs
    print(f"Fetching list of commits between {tag1} and {tag2}", file=sys.stderr)
    comparison = repo.compare(tag1, tag2)

    # get the pull requests for these commits
    print("Fetching pull requests", file=sys.stderr)
    unique_pulls = set()
    all_pulls = []
    for commit in comparison.commits:
        for pull in commit.get_pulls():
            # there can be multiple commits per PR if squash merge is not being used;
            # in that case just pick the first author
            if pull.number not in unique_pulls:
                unique_pulls.add(pull.number)
                all_pulls.append((pull, commit))

    # split the pulls into categories
    breaking = []
    bugs = []
    docs = []
    enhancements = []
    performance = []
    other = []

    # categorize the pull requests based on GitHub labels and Conventional Commits titles
    print("Categorizing pull requests", file=sys.stderr)
    for pull, commit in all_pulls:
        cc_type = ""
        cc_breaking = False
        parts = re.findall(r"^([a-z]+)(\([a-z0-9_-]+\))?(!)?:", pull.title)
        if len(parts) == 1:
            cc_type = parts[0][0]  # fix, feat, docs, chore, ...
            cc_breaking = parts[0][2] == "!"

        labels = [label.name for label in pull.labels]
        if "api change" in labels or cc_breaking:
            breaking.append((pull, commit))
        elif "bug" in labels or cc_type == "fix":
            bugs.append((pull, commit))
        elif "performance" in labels or cc_type == "perf":
            performance.append((pull, commit))
        elif "enhancement" in labels or cc_type == "feat":
            enhancements.append((pull, commit))
        elif "documentation" in labels or cc_type in ("docs", "doc"):
            docs.append((pull, commit))
        else:
            other.append((pull, commit))

    # produce the changelog content
    print("Generating changelog content", file=sys.stderr)

    print("""<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->
""")

    print(f"# Apache DataFusion Spark {version} Changelog\n")

    commit_count = len(git("log", "--pretty=oneline", f"{tag1}..{tag2}").splitlines())
    credits = git("shortlog", "-sn", f"{tag1}..{tag2}").rstrip()
    contributor_count = len(credits.splitlines())

    print(
        f"This release consists of {commit_count} commits from {contributor_count} contributors. "
        f"See credits at the end of this changelog for more information.\n"
    )

    print_pulls(repo_name, "Breaking changes", breaking)
    print_pulls(repo_name, "Performance related", performance)
    print_pulls(repo_name, "Implemented enhancements", enhancements)
    print_pulls(repo_name, "Fixed bugs", bugs)
    print_pulls(repo_name, "Documentation updates", docs)
    print_pulls(repo_name, "Other", other)

    print("## Credits\n")
    print(
        "Thank you to everyone who contributed to this release. Here is a breakdown of commits (PRs merged) "
        "per contributor.\n"
    )
    print("```")
    print(credits)
    print("```\n")

    print(
        "Thank you also to everyone who contributed in other ways such as filing issues, reviewing "
        "PRs, and providing feedback on this release."
    )


def cli(args=None):
    """Process command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("tag1", help="The previous commit or tag (e.g. v56.0.0)")
    parser.add_argument("tag2", help="The current commit or tag (e.g. HEAD)")
    parser.add_argument("version", help="The version number to include in the changelog")
    args = parser.parse_args(args)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        sys.exit("Please set a GitHub personal access token in the GITHUB_TOKEN environment variable")

    g = Github(auth=Auth.Token(token))
    repo = g.get_repo(PROJECT)
    generate_changelog(repo, PROJECT, args.tag1, args.tag2, args.version)


if __name__ == "__main__":
    cli()
