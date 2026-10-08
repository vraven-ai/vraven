#!/usr/bin/env python3
"""Download the raw VRAVEN cross-domain dataset suite without training models."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATASETS_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DATASETS_ROOT.parent


@dataclass(frozen=True)
class ArchiveDataset:
    key: str
    url: str
    archive_name: str = "source.zip"


ARCHIVE_DATASETS = (
    ArchiveDataset(
        "uci_har",
        "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    ),
    ArchiveDataset(
        "sms_spam",
        "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    ),
    ArchiveDataset(
        "wdbc",
        "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
    ),
    ArchiveDataset(
        "eeg_eye_state",
        "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip",
    ),
    ArchiveDataset(
        "ai4i_predictive_maintenance",
        "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
    ),
    ArchiveDataset(
        "dry_bean",
        "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    ),
    ArchiveDataset(
        "bank_marketing",
        "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
    ),
    ArchiveDataset(
        "wine_quality",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    ),
)


FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
FASHION_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion"
)

PROVENANCE = {
    "kronodroid": {
        "source_type": "existing_local_file",
        "source": "KR/kronodroid_combined.csv",
        "article_doi": "10.1016/j.cose.2021.102399",
        "redistribution_licence": "unverified_for_this_combined_file",
    },
    "fashion_mnist": {
        "source_type": "official_direct_files",
        "source": "https://github.com/zalandoresearch/fashion-mnist",
        "licence": "MIT",
        "citation": "arXiv:1708.07747",
    },
    "uci_har": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[0].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C54S4K",
    },
    "sms_spam": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[1].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C5CC84",
    },
    "wdbc": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[2].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C5DW2B",
    },
    "eeg_eye_state": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[3].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C57G7J",
    },
    "ai4i_predictive_maintenance": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[4].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C5HS5C",
    },
    "dry_bean": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[5].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C50S4B",
    },
    "bank_marketing": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[6].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C5K306",
    },
    "wine_quality": {
        "source_type": "official_uci_archive",
        "source": ARCHIVE_DATASETS[7].url,
        "licence": "CC BY 4.0",
        "doi": "10.24432/C56S3T",
    },
}


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        print(f"reuse    {destination.relative_to(REPOSITORY_ROOT)}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "VRAVEN-dataset-downloader/1.0"})
    print(f"download {url}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            member_target = (destination / member.filename).resolve()
            if not member_target.is_relative_to(destination_root):
                raise ValueError(f"unsafe archive member: {member.filename}")
        source.extractall(destination)


def include_kronodroid() -> None:
    source = REPOSITORY_ROOT / "KR" / "kronodroid_combined.csv"
    destination = DATASETS_ROOT / "kronodroid" / "raw" / source.name
    if not source.is_file():
        raise FileNotFoundError(f"existing KronoDroid source not found: {source}")
    if destination.exists():
        print(f"reuse    {destination.relative_to(REPOSITORY_ROOT)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        print(f"hardlink {destination.relative_to(REPOSITORY_ROOT)}")
    except OSError:
        shutil.copy2(source, destination)
        print(f"copy     {destination.relative_to(REPOSITORY_ROOT)}")


def download_archives() -> None:
    for dataset in ARCHIVE_DATASETS:
        raw = DATASETS_ROOT / dataset.key / "raw"
        archive = raw / dataset.archive_name
        download(dataset.url, archive)
        safe_extract(archive, raw)
        for nested_archive in sorted(raw.rglob("*.zip")):
            if nested_archive.resolve() != archive.resolve():
                safe_extract(nested_archive, nested_archive.parent)


def download_fashion_mnist() -> None:
    raw = DATASETS_ROOT / "fashion_mnist" / "raw"
    for filename, expected_md5 in FASHION_FILES.items():
        destination = raw / filename
        download(f"{FASHION_BASE_URL}/{filename}", destination)
        actual_md5 = digest(destination, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(
                f"Fashion-MNIST checksum mismatch for {filename}: "
                f"expected {expected_md5}, received {actual_md5}"
            )


def write_manifest() -> None:
    records = []
    for dataset_dir in sorted(path for path in DATASETS_ROOT.iterdir() if path.is_dir()):
        raw = dataset_dir / "raw"
        files = []
        if raw.is_dir():
            for path in sorted(candidate for candidate in raw.rglob("*") if candidate.is_file()):
                files.append(
                    {
                        "path": path.relative_to(DATASETS_ROOT).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": digest(path),
                    }
                )
        records.append(
            {
                "dataset": dataset_dir.name,
                "provenance": PROVENANCE[dataset_dir.name],
                "files": files,
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "datasets": records,
    }
    target = DATASETS_ROOT / "DOWNLOAD_MANIFEST.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest {target.relative_to(REPOSITORY_ROOT)}")


def main() -> None:
    include_kronodroid()
    download_fashion_mnist()
    download_archives()
    write_manifest()


if __name__ == "__main__":
    main()
