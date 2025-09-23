#!/usr/bin/env python

import os
import importlib
import shutil


def test_surusc_cache_set_on_import(tmp_path, monkeypatch):
    # Ensure env is clean for this test
    monkeypatch.delenv("SURUS_CACHE", raising=False)

    # Point HOME to a temp directory to avoid touching real user cache
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))

    # Reload module to apply new env
    if "sulat" in list(importlib.sys.modules.keys()):
        importlib.invalidate_caches()
        importlib.reload(importlib.import_module("sulat"))

    # Fresh import
    import sulat  # noqa: F401

    expected = os.path.abspath(os.path.expanduser("~/.cache/surus"))
    # Ensure env var is set
    assert os.environ.get("SURUS_CACHE") == expected
    # Ensure directory exists
    assert os.path.isdir(expected)
