from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from app.config.settings import get_settings

M5_ARCHIVE_URL = (
    "https://zenodo.org/records/12636070/files/"
    "m5-forecasting-accuracy.zip?download=1"
)
M5_ARCHIVE_MD5 = "86f57416a314197f40a17cc6fc60cbb4"
REQUIRED_FILES = (
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
)


def calculate_md5(file_path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    request = urllib.request.Request(
        M5_ARCHIVE_URL,
        headers={"User-Agent": "m5-demand-forecasting-platform/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)


def extract_required_files(archive_path: Path, data_dir: Path, force: bool) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        missing_members = set(REQUIRED_FILES) - set(archive.namelist())
        if missing_members:
            raise ValueError(
                f"M5 archive is missing required files: {sorted(missing_members)}"
            )

        existing_targets = [
            data_dir / file_name
            for file_name in REQUIRED_FILES
            if (data_dir / file_name).exists()
        ]
        if existing_targets and not force:
            raise FileExistsError(
                "Refusing to overwrite existing raw files: "
                f"{[str(path) for path in existing_targets]}; rerun with --force"
            )

        for file_name in REQUIRED_FILES:
            target = data_dir / file_name
            temp_target = target.with_name(f".{target.name}.tmp")
            try:
                with (
                    archive.open(file_name) as source,
                    temp_target.open("wb") as output,
                ):
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                temp_target.replace(target)
            finally:
                temp_target.unlink(missing_ok=True)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Download and verify the public M5 forecasting dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.raw_data_dir,
        help="Destination for the three raw CSV files",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing M5 CSV files"
    )
    args = parser.parse_args()

    expected_paths = [args.data_dir / name for name in REQUIRED_FILES]
    if all(path.exists() for path in expected_paths) and not args.force:
        print(f"M5 data is already present in {args.data_dir}")
        return

    with tempfile.TemporaryDirectory(prefix="m5-download-") as temp_dir:
        archive_path = Path(temp_dir) / "m5-forecasting-accuracy.zip"
        print(f"Downloading M5 archive to {archive_path}")
        download_archive(archive_path)

        actual_md5 = calculate_md5(archive_path)
        if actual_md5 != M5_ARCHIVE_MD5:
            raise ValueError(
                f"M5 archive checksum mismatch: {actual_md5} != {M5_ARCHIVE_MD5}"
            )

        extract_required_files(archive_path, args.data_dir, force=args.force)

    print(f"Verified M5 files installed in {args.data_dir}")


if __name__ == "__main__":
    main()
