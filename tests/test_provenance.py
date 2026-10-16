# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

from pathlib import Path

from class_hierarchy_cl.provenance import main_revision, pinned_main_revision

ROOT = Path(__file__).resolve().parents[1]


def test_pin_is_a_full_sha():
    pin = pinned_main_revision(ROOT / "pyproject.toml")
    assert pin is not None and len(pin) == 40


def test_installed_main_matches_pin_when_installed_from_git():
    rev = main_revision()
    if (
        rev is None
    ):  # editable / local install: nothing to compare, the run must record None explicitly
        return
    assert rev == pinned_main_revision(ROOT / "pyproject.toml")
