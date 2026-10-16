# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Which revision of incremental-feature-cl (`main`) is installed, and of this package."""

from __future__ import annotations

import json
import re
import subprocess
from importlib import metadata
from pathlib import Path

MAIN_DIST = "incremental-feature-cl"
_SHA = re.compile(r"^[0-9a-f]{40}$")


def main_revision() -> str | None:
    """Commit SHA pip recorded for the git install of incremental-feature-cl (None if not a VCS install)."""
    try:
        raw = metadata.distribution(MAIN_DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    sha = json.loads(raw).get("vcs_info", {}).get("commit_id")
    return sha if sha and _SHA.match(sha) else None


def pinned_main_revision(pyproject: Path) -> str | None:
    """The SHA this package's pyproject pins."""
    m = re.search(r"incremental_feature_cl\.git@([0-9a-f]{40})", pyproject.read_text())
    return m.group(1) if m else None


def code_revision(path: Path | None = None) -> str | None:
    """Git HEAD of this package's checkout, or None outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
