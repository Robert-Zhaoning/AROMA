from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "pyproject.toml",
    "docs/README.md",
    "docs/PROJECT_STATUS.md",
    "docs/ENVIRONMENT.md",
    "docs/DATA.md",
    "docs/REPOSITORY_STRUCTURE.md",
    "docs/EXPERIMENT_LEDGER.md",
    "docs/RESULTS_CANONICAL.md",
    "docs/ARTIFACT_MANIFEST.md",
    "docs/REPRODUCIBILITY.md",
    "docs/AROMA2_CSA_PROTOCOL_v1_1.md",
    "docs/AROMA2_CSA_V4_FINAL_RESULT_v1.md",
    "configs/tracked_outputs_allowlist.txt",
]


SECRET_PATTERNS = [
    (
        "OpenSSH private key",
        re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"),
    ),
    (
        "generic private key",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"),
    ),
    (
        "Hugging Face token",
        re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "GitHub classic token",
        re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    ),
    (
        "GitHub fine-grained token",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    ),
]


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def tracked_files() -> list[str]:
    out = run_git("ls-files", "-z")
    return [x for x in out.split("\0") if x]


def check_required() -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            errors.append(f"missing required file: {rel}")

    return errors


def check_output_allowlist() -> list[str]:
    errors: list[str] = []

    allowlist_path = ROOT / "configs/tracked_outputs_allowlist.txt"

    if not allowlist_path.exists():
        return ["missing tracked-output allowlist"]

    allowed = {
        line.strip()
        for line in allowlist_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    current = {
        path
        for path in tracked_files()
        if path.startswith("outputs/")
    }

    unexpected = sorted(current - allowed)

    if unexpected:
        errors.append(
            "new tracked outputs are not in the frozen allowlist:\n  "
            + "\n  ".join(unexpected)
        )

    return errors


def check_large_files() -> list[str]:
    errors: list[str] = []

    limit = 50 * 1024 * 1024

    for rel in tracked_files():
        path = ROOT / rel

        if not path.is_file():
            continue

        size = path.stat().st_size

        if size > limit:
            errors.append(
                f"tracked file exceeds 50 MiB: {rel} "
                f"({size / 1024 / 1024:.1f} MiB)"
            )

    return errors


def check_json() -> list[str]:
    errors: list[str] = []

    for rel in tracked_files():
        if not rel.endswith(".json"):
            continue

        path = ROOT / rel

        if not path.is_file():
            continue

        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON: {rel}: {exc}")

    return errors


def check_secrets() -> list[str]:
    errors: list[str] = []

    max_scan = 2 * 1024 * 1024

    for rel in tracked_files():
        path = ROOT / rel

        if not path.is_file():
            continue

        if path.stat().st_size > max_scan:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"possible {name} in tracked file: {rel}"
                )

    return errors


def main() -> int:
    checks = [
        ("required files", check_required),
        ("tracked outputs", check_output_allowlist),
        ("large files", check_large_files),
        ("JSON", check_json),
        ("secret patterns", check_secrets),
    ]

    all_errors: list[str] = []

    print("=" * 72)
    print("AROMA REPOSITORY INTEGRITY CHECK")
    print("=" * 72)

    for name, fn in checks:
        errors = fn()

        if errors:
            print(f"[FAIL] {name}")

            for error in errors:
                print(f"  - {error}")

            all_errors.extend(errors)

        else:
            print(f"[ OK ] {name}")

    print("=" * 72)

    if all_errors:
        print(f"FAILED: {len(all_errors)} issue(s)")
        return 1

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
