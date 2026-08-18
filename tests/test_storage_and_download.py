from __future__ import annotations

import hashlib
import zipfile

import pandas as pd
import pytest

from app.data.storage import load_parquet, save_parquet
from scripts.download_m5_data import (
    REQUIRED_FILES,
    calculate_md5,
    extract_required_files,
)


def test_parquet_round_trip_is_atomic(tmp_path) -> None:
    output = tmp_path / "nested" / "table.parquet"
    expected = pd.DataFrame({"value": [1, 2, 3]})

    save_parquet(expected, output)
    actual = load_parquet(output)

    pd.testing.assert_frame_equal(actual, expected)
    assert not (output.parent / f".{output.name}.tmp").exists()


def test_load_parquet_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Parquet file not found"):
        load_parquet(tmp_path / "missing.parquet")


def test_download_checksum_and_safe_required_extraction(tmp_path) -> None:
    archive_path = tmp_path / "m5.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in REQUIRED_FILES:
            archive.writestr(name, f"content for {name}")
        archive.writestr("../unsafe.txt", "must not be extracted")

    assert (
        calculate_md5(archive_path)
        == hashlib.md5(archive_path.read_bytes(), usedforsecurity=False).hexdigest()
    )

    output_dir = tmp_path / "raw"
    extract_required_files(archive_path, output_dir, force=False)

    assert all((output_dir / name).exists() for name in REQUIRED_FILES)
    assert not (tmp_path / "unsafe.txt").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extract_required_files(archive_path, output_dir, force=False)


def test_download_rejects_incomplete_archive(tmp_path) -> None:
    archive_path = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("calendar.csv", "date,d")

    with pytest.raises(ValueError, match="missing required files"):
        extract_required_files(archive_path, tmp_path / "raw", force=False)


def test_download_preflights_existing_files_before_extraction(tmp_path) -> None:
    archive_path = tmp_path / "m5.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in REQUIRED_FILES:
            archive.writestr(name, f"content for {name}")

    output_dir = tmp_path / "raw"
    output_dir.mkdir()
    existing_path = output_dir / REQUIRED_FILES[1]
    existing_path.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        extract_required_files(archive_path, output_dir, force=False)

    assert existing_path.read_text(encoding="utf-8") == "keep me"
    assert not (output_dir / REQUIRED_FILES[0]).exists()
