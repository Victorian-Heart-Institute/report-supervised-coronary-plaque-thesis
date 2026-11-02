"""Redacted privacy checks for staged files or every reachable Git commit.

Complement this guard with a dedicated secret scanner and human review. Passing
does not prove absence of every possible confidential fact or credential.
"""

import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess

PATTERNS = {
    "private_absolute_path": r"/(?:Users|home|fs\d+|scratch\d*)/[^\s\"\'<>]+",
    "internal_project_path": r"(?:research/CORA-run|outputs/experiment_results)/[^\s\"\'<>]+",
    "clinical_scan_identifier": r"\b\d{5,10}-(?:t\d+|\d+)\b",
    "github_credential": r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{12,}",
    "private_key": r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----",
    "aws_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "api_key": r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}",
}
FORBIDDEN_EXTENSIONS = {
    ".npz",
    ".npy",
    ".nii",
    ".gz",
    ".dcm",
    ".pth",
    ".pt",
    ".ckpt",
    ".pem",
    ".key",
    ".docx",
}


def git(*args):
    return subprocess.check_output(["git", *args])


def scan_blob(path, blob, revision):
    findings = []
    p = PurePosixPath(path)
    if p.suffix.lower() in FORBIDDEN_EXTENSIONS or p.name == ".env":
        findings.append(
            {"revision": revision, "file": path, "rule": "restricted_file_type"}
        )
    try:
        content = blob.decode("utf-8")
    except UnicodeDecodeError:
        return findings + [
            {"revision": revision, "file": path, "rule": "binary_file_requires_review"}
        ]
    for rule, pattern in PATTERNS.items():
        for number, line in enumerate(content.splitlines(), 1):
            if re.search(pattern, line):
                findings.append(
                    {"revision": revision, "file": path, "line": number, "rule": rule}
                )
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-history", action="store_true")
    args = parser.parse_args()
    findings = []
    scanned = 0
    if args.all_history:
        commits = git("rev-list", "--all").decode().splitlines()
        for commit in commits:
            emails = (
                git("show", "-s", "--format=%ae%n%ce", commit).decode().splitlines()
            )
            if not all(e.endswith("@users.noreply.github.com") for e in emails):
                findings.append(
                    {"revision": commit[:12], "rule": "non_noreply_git_identity"}
                )
            paths = (
                git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
            )
            for path in filter(None, paths):
                findings += scan_blob(
                    path, git("show", commit + ":" + path), commit[:12]
                )
                scanned += 1
    else:
        identity = git("var", "GIT_AUTHOR_IDENT").decode()
        if not re.search(r"<[^<>]+@users\.noreply\.github\.com>", identity):
            findings.append({"revision": "staged", "rule": "non_noreply_git_identity"})
        paths = (
            git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
            .decode()
            .split("\0")
        )
        for path in filter(None, paths):
            findings += scan_blob(path, git("show", ":" + path), "staged")
            scanned += 1
    print(
        json.dumps({"scanned_file_versions": scanned, "findings": findings}, indent=2)
    )
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
