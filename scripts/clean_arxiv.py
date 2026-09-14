#!/usr/bin/env python3
"""Simple arXiv download and cleaning script."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_RAW_FILE = PROJECT_ROOT / "data" / "raw" / "arxiv-metadata-oai-snapshot.zip"
FULL_CLEAN_FILE = PROJECT_ROOT / "data" / "processed" / "arxiv_metadata_clean.csv"

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"

KEEP_COLUMNS = [
    "paper_id",
    "title",
    "abstract",
    "submitted_year",
    "submitted_date",
    "categories",
    "primary_category",
    "authors",
    "doi",
    "journal_ref",
    "license",
    "update_date",
]

IMPORTANT_COLUMNS = ["paper_id", "title", "abstract", "submitted_year", "categories"]


def clean_text(value) -> str:
    """Turn missing values into blanks and collapse extra spaces."""

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).split())


def pick_first(row, names: list[str]):
    """Pick the first useful value from a row."""

    for name in names:
        value = row.get(name, "")
        if clean_text(value):
            return value
    return ""


def download_full_snapshot(output_file: Path, force: bool = False) -> None:
    """Download the full Kaggle arXiv metadata snapshot."""

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists() and not force:
        print(f"Using existing full snapshot: {output_file}")
        return

    partial_file = output_file.with_suffix(output_file.suffix + ".partial")

    with requests.get(KAGGLE_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))
        downloaded_bytes = 0

        with partial_file.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded_bytes += len(chunk)

                downloaded_mb = downloaded_bytes / 1_000_000
                if total_bytes:
                    total_mb = total_bytes / 1_000_000
                    print(f"\rDownloaded {downloaded_mb:.1f} of {total_mb:.1f} MB", end="")
                else:
                    print(f"\rDownloaded {downloaded_mb:.1f} MB", end="")

    print()
    partial_file.replace(output_file)
    print(f"Saved full snapshot to {output_file}")


def read_raw_metadata(raw_file: Path, chunksize: int):
    """Read a JSONL file or the Kaggle zip file in small pieces."""

    if raw_file.suffix == ".zip":
        with zipfile.ZipFile(raw_file) as archive:
            json_files = [name for name in archive.namelist() if name.endswith(".json")]
            if not json_files:
                raise FileNotFoundError(f"No JSON file found inside {raw_file}")

            with archive.open(json_files[0]) as metadata_file:
                yield from pd.read_json(metadata_file, lines=True, chunksize=chunksize)
    else:
        yield from pd.read_json(raw_file, lines=True, chunksize=chunksize)


def submitted_date(row) -> str:
    """Use the first version date when available; otherwise use a simple date field."""

    versions = row.get("versions", "")
    if isinstance(versions, list) and versions:
        first_version = versions[0]
        if isinstance(first_version, dict):
            return clean_text(first_version.get("created", ""))

    return clean_text(pick_first(row, ["created", "published", "updated", "update_date"]))


def authors(row) -> str:
    """Use the author string when available; otherwise build names from parsed authors."""

    author_text = clean_text(row.get("authors", ""))
    if author_text:
        return author_text

    parsed_authors = row.get("authors_parsed", "")
    if not isinstance(parsed_authors, list):
        return ""

    names = []
    for author in parsed_authors:
        if not isinstance(author, list):
            continue

        last_name = author[0] if len(author) > 0 else ""
        first_name = author[1] if len(author) > 1 else ""
        full_name = clean_text(f"{first_name} {last_name}")
        if full_name:
            names.append(full_name)

    return ", ".join(names)


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Select the columns we need and make them easier to analyze."""

    clean = pd.DataFrame()

    clean["paper_id"] = chunk.apply(
        lambda row: clean_text(pick_first(row, ["id", "paper_id", "identifier"])),
        axis=1,
    )
    clean["paper_id"] = clean["paper_id"].str.replace(r"v\d+$", "", regex=True)
    clean["paper_id"] = clean["paper_id"].str.replace("oai:arXiv.org:", "", regex=False)

    clean["title"] = chunk.apply(lambda row: clean_text(row.get("title", "")), axis=1)
    clean["abstract"] = chunk.apply(
        lambda row: clean_text(pick_first(row, ["abstract", "summary"])),
        axis=1,
    )
    clean["categories"] = chunk.apply(lambda row: clean_text(row.get("categories", "")), axis=1)
    clean["primary_category"] = clean["categories"].str.split().str[0].fillna("")
    clean["authors"] = chunk.apply(authors, axis=1)
    clean["doi"] = chunk.apply(lambda row: clean_text(row.get("doi", "")), axis=1)
    clean["journal_ref"] = chunk.apply(
        lambda row: clean_text(pick_first(row, ["journal-ref", "journal_ref"])),
        axis=1,
    )
    clean["license"] = chunk.apply(lambda row: clean_text(row.get("license", "")), axis=1)
    clean["update_date"] = chunk.apply(
        lambda row: clean_text(pick_first(row, ["update_date", "updated"])),
        axis=1,
    )

    date_text = chunk.apply(submitted_date, axis=1)
    try:
        dates = pd.to_datetime(date_text, errors="coerce", utc=True, format="mixed")
    except TypeError:
        dates = pd.to_datetime(date_text, errors="coerce", utc=True)

    clean["submitted_date"] = dates.dt.strftime("%Y-%m-%d").fillna("")
    clean["submitted_year"] = dates.dt.year.astype("Int64").astype(str).replace("<NA>", "")

    return clean[KEEP_COLUMNS]


def clean_metadata(raw_file: Path, clean_file: Path, chunksize: int = 100_000) -> None:
    """Create one clean CSV file from the raw arXiv metadata."""

    clean_file.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    first_chunk = True
    summary = {
        "raw_records": 0,
        "clean_records": 0,
        "dropped_duplicates": 0,
        "dropped_missing_important_values": 0,
        "important_columns": IMPORTANT_COLUMNS,
        "kept_columns": KEEP_COLUMNS,
    }

    for raw_chunk in read_raw_metadata(raw_file, chunksize):
        summary["raw_records"] += len(raw_chunk)
        clean_chunk = prepare_chunk(raw_chunk)

        before_missing_drop = len(clean_chunk)
        clean_chunk = clean_chunk[~clean_chunk[IMPORTANT_COLUMNS].eq("").any(axis=1)]
        summary["dropped_missing_important_values"] += before_missing_drop - len(clean_chunk)

        before_duplicate_drop = len(clean_chunk)
        clean_chunk = clean_chunk.drop_duplicates(subset="paper_id", keep="first")
        clean_chunk = clean_chunk[~clean_chunk["paper_id"].isin(seen_ids)]
        summary["dropped_duplicates"] += before_duplicate_drop - len(clean_chunk)

        seen_ids.update(clean_chunk["paper_id"])
        summary["clean_records"] += len(clean_chunk)

        clean_chunk.to_csv(
            clean_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False

    summary_file = clean_file.with_suffix(clean_file.suffix + ".summary.json")
    summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved clean data to {clean_file}")
    print(f"Saved cleaning summary to {summary_file}")
    print(json.dumps(summary, indent=2))


def run_full(
    raw_file: Path = FULL_RAW_FILE,
    clean_file: Path = FULL_CLEAN_FILE,
    force_download: bool = False,
    chunksize: int = 100_000,
) -> None:
    """Download the full snapshot, then clean it."""

    download_full_snapshot(Path(raw_file), force=force_download)
    clean_metadata(Path(raw_file), Path(clean_file), chunksize=chunksize)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean the full arXiv metadata.")
    parser.add_argument("--raw-output", type=Path, default=FULL_RAW_FILE)
    parser.add_argument("--clean-output", type=Path, default=FULL_CLEAN_FILE)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--chunksize", type=int, default=100_000)

    args = parser.parse_args()

    run_full(
        raw_file=args.raw_output,
        clean_file=args.clean_output,
        force_download=args.force_download,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
