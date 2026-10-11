"""Publish one tested branch or version tag from the dedicated extension repository."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path

VERSION_TAG = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def publication_ref(environment: Mapping[str, str]) -> str:
    if environment.get("CI_PIPELINE_SOURCE") != "push":
        raise ValueError("Publication requires a push pipeline")
    if environment.get("CI_COMMIT_REF_PROTECTED") != "true":
        raise ValueError("Publication requires a protected ref")
    tag = environment.get("CI_COMMIT_TAG", "")
    if tag:
        if not VERSION_TAG.fullmatch(tag):
            raise ValueError("Publication requires a version tag such as v1.2.3")
        return f"refs/tags/{tag}"
    default_branch = environment.get("CI_DEFAULT_BRANCH", "")
    if not default_branch or environment.get("CI_COMMIT_BRANCH") != default_branch:
        raise ValueError("Only the default branch can be published")
    return f"refs/heads/{default_branch}"


def publish(repository: Path, remote: str, ref: str, commit: str) -> str:
    """Push one explicit ref without force, then verify the remote object ID."""
    git(repository, "check-ref-format", ref)
    if git(repository, "rev-parse", "HEAD") != commit:
        raise ValueError("The checkout does not match the tested CI commit")
    source = commit
    if ref.startswith("refs/tags/"):
        if git(repository, "rev-parse", f"{ref}^{{commit}}") != commit:
            raise ValueError("The version tag does not match the tested CI commit")
        git(repository, "merge-base", "--is-ancestor", commit, "refs/remotes/origin/main")
        source = git(repository, "rev-parse", ref)
    git(repository, "push", "--atomic", "--porcelain", remote, f"{source}:{ref}")
    actual = git(repository, "ls-remote", "--refs", remote, ref)
    if actual != f"{source}\t{ref}":
        raise RuntimeError("Published ref verification failed")
    return source


def configure_ssh(environment: Mapping[str, str]) -> str:
    repository = environment.get("GITHUB_REPOSITORY", "")
    if not GITHUB_REPOSITORY.fullmatch(repository):
        raise ValueError("GITHUB_REPOSITORY must be an owner/repository name")
    key = Path(environment["GITHUB_SSH_PRIVATE_KEY"])
    known_hosts = Path(environment["GITHUB_SSH_KNOWN_HOSTS"])
    if not key.is_file() or not known_hosts.is_file():
        raise ValueError("GitHub SSH credentials must be GitLab file variables")
    key.chmod(0o600)
    os.environ["GIT_SSH_COMMAND"] = shlex.join(
        [
            "ssh",
            "-i",
            str(key.resolve()),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={known_hosts.resolve()}",
        ]
    )
    os.environ["GIT_TERMINAL_PROMPT"] = "0"
    return f"git@github.com:{repository}.git"


def main() -> None:
    ref = publication_ref(os.environ)
    remote = configure_ssh(os.environ)
    repository = Path(__file__).resolve().parents[2]
    commit = os.environ["CI_COMMIT_SHA"]
    published = publish(repository, remote, ref, commit)
    print(f"Published and verified {ref} at {published}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr.strip()) from error
