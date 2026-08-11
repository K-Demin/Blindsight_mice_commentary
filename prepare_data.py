#!/usr/bin/env python3
"""Prepare analysis inputs from the authors' official source-data archive.

No source data are distributed with this code. The script downloads the data
from existing repository mentioned in the original manuscript.

Seven required tables are extracted from modern GraphPad ``.prism`` files.
The two Figure 2 tables are stored in a legacy binary ``.pzf`` file and must be
exported to CSV with GraphPad Prism before they can be copied and validated by
this script.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath


OFFICIAL_ARCHIVE_URL = "https://iam.science/data/Bhatla_et_al_2026_DATA.zip"

# (archive member, exact Prism data-sheet title, local output filename)
PRISM_TABLES = [
    (
        "Figure 1 - 3-choice.prism",
        "Fig1E - V1 suction - 3-choice choices",
        "Fig1E.csv",
    ),
    (
        "Figure 5 - V1 ibo + musc, dLGN musc.prism",
        "Fig 5E - V1 Ibo (<50% Hipp killed) - 3-choice choices",
        "Fig 5E.csv",
    ),
    (
        "Figure 5 - V1 ibo + musc, dLGN musc.prism",
        "Fig 5J - V1 musc - All 3-choice choices",
        "Fig 5J.csv",
    ),
    (
        "Figure 5 - V1 ibo + musc, dLGN musc.prism",
        "Fig 5N - dLGN musc - All 3-choice choices",
        "Fig 5N.csv",
    ),
    (
        "Figure 6 - Hipp+V1 ibo, Hipp ibo, Hipp musc.prism",
        "Fig 6E - V1 Ibo (>50% Hipp killed) - All 3-choice choices",
        "Fig 6E.csv",
    ),
    (
        "Figure 6 - Hipp+V1 ibo, Hipp ibo, Hipp musc.prism",
        "Fig 6K - Hipp Ibo - 3-choice choices",
        "Fig 6K.csv",
    ),
    (
        "Figure 6 - Hipp+V1 ibo, Hipp ibo, Hipp musc.prism",
        "Fig 6P - Hipp musc - All 3-choice choices",
        "Fig 6P.csv",
    ),
]

FIGURE_2_ARCHIVE_MEMBER = "Figure 2 - salience.pzf"
FIGURE_2_EXPORTS = {
    "Fig2D.csv": "Fig2D: R Opa 3-6% - 3-choice choices - pre/post R dim",
    "Fig2E.csv": "Fig2E: R Opa 100% - 3-choice choices - pre/post V1 suction",
}

EXPECTED_INPUTS = [
    "Fig1E.csv",
    "Fig2D.csv",
    "Fig2E.csv",
    "Fig 5E.csv",
    "Fig 5J.csv",
    "Fig 5N.csv",
    "Fig 6E.csv",
    "Fig 6K.csv",
    "Fig 6P.csv",
]


def archive_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if PurePosixPath(name).name == basename
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one archive member named {basename!r}; found {matches}"
        )
    return matches[0]


def title_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("string", "")
        if isinstance(text, str):
            return text
    return ""


def extract_prism_table(
    outer_archive: zipfile.ZipFile,
    prism_member_name: str,
    sheet_title: str,
    output_path: Path,
) -> None:
    member = archive_member(outer_archive, prism_member_name)
    prism_bytes = io.BytesIO(outer_archive.read(member))
    if not zipfile.is_zipfile(prism_bytes):
        raise ValueError(f"{prism_member_name!r} is not a readable modern Prism archive")
    prism_bytes.seek(0)

    with zipfile.ZipFile(prism_bytes) as prism:
        matched_sheet: dict[str, object] | None = None
        for name in prism.namelist():
            if not name.startswith("data/sheets/") or not name.endswith("/sheet.json"):
                continue
            sheet = json.loads(prism.read(name))
            if sheet.get("title") == sheet_title:
                matched_sheet = sheet
                break

        if matched_sheet is None:
            raise ValueError(
                f"Could not find Prism sheet {sheet_title!r} in {prism_member_name!r}"
            )

        table = matched_sheet.get("table")
        if not isinstance(table, dict):
            raise ValueError(f"Sheet {sheet_title!r} does not contain a data table")
        table_uid = table.get("uid")
        dataset_uids = table.get("dataSets")
        if not isinstance(table_uid, str) or not isinstance(dataset_uids, list):
            raise ValueError(f"Sheet {sheet_title!r} has an unsupported table structure")

        headers: list[str] = []
        for dataset_uid in dataset_uids:
            dataset = json.loads(prism.read(f"data/sets/{dataset_uid}.json"))
            headers.append(title_text(dataset.get("title")))

        raw_csv = prism.read(f"data/tables/{table_uid}/data.csv").decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(raw_csv)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["", *headers])
        writer.writerows(rows)


def numeric_cell_count(row: list[str]) -> int:
    count = 0
    for cell in row:
        try:
            float(cell.strip())
        except ValueError:
            continue
        count += 1
    return count


def validate_input_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    if numeric_cell_count(rows[0]) >= 36:
        raise ValueError(
            f"{path} appears to have no header row. Export the Prism table with column titles."
        )
    usable = sum(numeric_cell_count(row) >= 36 for row in rows[1:])
    if usable == 0:
        raise ValueError(f"{path} contains no rows with the required 36 numeric values")
    return usable


def copy_figure_2_export(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    validate_input_csv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help=f"Downloaded official archive ({OFFICIAL_ARCHIVE_URL})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Data",
        help="Destination for the nine analysis CSVs (default: ./Data)",
    )
    parser.add_argument(
        "--fig2d-csv",
        type=Path,
        help="CSV exported from the exact Fig2D Prism sheet documented in README.md",
    )
    parser.add_argument(
        "--fig2e-csv",
        type=Path,
        help="CSV exported from the exact Fig2E Prism sheet documented in README.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    if bool(args.fig2d_csv) != bool(args.fig2e_csv):
        raise ValueError("Provide both --fig2d-csv and --fig2e-csv, or neither")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as outer_archive:
        # Confirm that the legacy Figure 2 source is present even though it
        # cannot be decoded by this standard-library-only helper.
        archive_member(outer_archive, FIGURE_2_ARCHIVE_MEMBER)
        for prism_member, sheet_title, output_name in PRISM_TABLES:
            destination = args.data_dir / output_name
            extract_prism_table(
                outer_archive,
                prism_member,
                sheet_title,
                destination,
            )
            print(f"prepared {destination} ({validate_input_csv(destination)} usable rows)")

    if args.fig2d_csv and args.fig2e_csv:
        for source, output_name in [
            (args.fig2d_csv, "Fig2D.csv"),
            (args.fig2e_csv, "Fig2E.csv"),
        ]:
            destination = args.data_dir / output_name
            copy_figure_2_export(source, destination)
            print(f"prepared {destination} ({validate_input_csv(destination)} usable rows)")
    else:
        print("\nSeven of nine inputs were prepared.")
        print("Export the following two sheets from 'Figure 2 - salience.pzf':")
        for output_name, sheet_title in FIGURE_2_EXPORTS.items():
            print(f"  {sheet_title!r} -> {output_name}")
        print("Then rerun this command with --fig2d-csv and --fig2e-csv.")

    missing = [name for name in EXPECTED_INPUTS if not (args.data_dir / name).exists()]
    if missing:
        raise SystemExit(f"Data preparation is incomplete; missing: {', '.join(missing)}")

    print("\nAll nine required input CSVs are present and structurally valid.")


if __name__ == "__main__":
    main()
