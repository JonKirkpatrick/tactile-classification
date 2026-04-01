#!/usr/bin/env python3
"""Compatibility wrapper for CSV-to-hex conversion.

Prefer running export_weights.py, which now handles both CSV export and combined
hex generation in one pass.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.weights.export_weights import collect_ordered_csv_paths, convert_csv_files_to_hex  # noqa: E402


def main():
    csv_dir = ROOT / "artifacts" / "weights" / "csv"
    csv_files = collect_ordered_csv_paths(csv_dir)
    if not csv_files:
        print("No expected CSV files found. Run export_weights.py first.")
        return

    out_hex = ROOT / "artifacts" / "weights" / "hex" / "weights_combined.hex"
    convert_csv_files_to_hex(csv_files, str(out_hex), clamp=True, verbose=True)


if __name__ == "__main__":
    main()

