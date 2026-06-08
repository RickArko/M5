from __future__ import annotations

from m5.cli import _RAW_REQUIRED_FILES, _missing_raw_files


def test_missing_raw_files_detects_empty_directory(tmp_path) -> None:
    assert _missing_raw_files(tmp_path) == list(_RAW_REQUIRED_FILES)


def test_missing_raw_files_requires_non_empty_csvs(tmp_path) -> None:
    for name in _RAW_REQUIRED_FILES:
        (tmp_path / name).touch()

    assert _missing_raw_files(tmp_path) == list(_RAW_REQUIRED_FILES)

    for name in _RAW_REQUIRED_FILES:
        (tmp_path / name).write_text("header\nvalue\n")

    assert _missing_raw_files(tmp_path) == []
