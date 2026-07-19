"""Shared test hygiene for the ablation suite."""

from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _rederive_fast_scan_latch():
    """Re-derive gdn3.kmd2_native._FAST_SCAN from the environment per test.

    The module latches GDN3_FAST_SCAN at first import, so whichever test
    happened to trigger that import (possibly under monkeypatch.setenv) used
    to leak the flag into every later test in the process.  Production is
    unaffected: the launcher exports the flag before Python starts.
    """
    module = sys.modules.get("gdn3.kmd2_native")
    if module is not None:
        module._FAST_SCAN = os.environ.get("GDN3_FAST_SCAN", "0") == "1"
    yield
