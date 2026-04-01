#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path

try:
    from _bootstrap import ensure_modeling_root_on_path
except ImportError:
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _bootstrap import ensure_modeling_root_on_path

MODEL_ROOT = ensure_modeling_root_on_path()
DEFAULT_CHECKPOINT = str(MODEL_ROOT / "artifacts" / "models" / "int8_model.pt")
DEFAULT_CSV_DIR = str(MODEL_ROOT / "artifacts" / "weights" / "csv")
DEFAULT_HEX_OUT = str(MODEL_ROOT / "artifacts" / "weights" / "hex" / "weights_combined.hex")

try:
    import torch
except ModuleNotFoundError:
    torch = None


def next_power_of_two(n):
    return 1 if n <= 1 else 2 ** math.ceil(math.log2(n))


def to_int8_safe(x, clamp=True):
    val = int(float(x))
    if -128 <= val <= 127:
        return val, False, None
    if not clamp:
        raise ValueError(f"value {val} out of int8 range")
    val_clamped = max(-128, min(127, val))
    return val_clamped, True, f"clamped from {val}"


def _normalize_state_dict_keys(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(loaded):
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        return loaded["state_dict"]
    return loaded


def infer_spec_from_state_dict(state_dict, temporal_length=128):
    block_ids = sorted(
        {
            int(m.group(1))
            for key in state_dict.keys()
            for m in [re.match(r"blocks\.(\d+)\.dw\.weight$", key)]
            if m
        }
    )
    if not block_ids:
        raise ValueError("Could not infer block structure from state_dict keys.")

    blocks = []
    for i in block_ids:
        dw_key = f"blocks.{i}.dw.weight"
        pw_key = f"blocks.{i}.pw.weight"
        if dw_key not in state_dict or pw_key not in state_dict:
            raise KeyError(f"Missing expected keys: {dw_key} and/or {pw_key}")

        dw_weight = state_dict[dw_key]
        pw_weight = state_dict[pw_key]

        if dw_weight.ndim != 3:
            raise ValueError(f"Unexpected shape for {dw_key}: {tuple(dw_weight.shape)}")
        if pw_weight.ndim != 3:
            raise ValueError(f"Unexpected shape for {pw_key}: {tuple(pw_weight.shape)}")

        kernel = int(dw_weight.shape[2])
        out_channels = int(pw_weight.shape[0])
        blocks.append((kernel, out_channels, None))

    sensor_channels = int(state_dict[f"blocks.{block_ids[0]}.dw.weight"].shape[0])
    if "classifier.weight" not in state_dict:
        raise KeyError("Missing classifier.weight in checkpoint")
    num_classes = int(state_dict["classifier.weight"].shape[0])

    return (sensor_channels, temporal_length, blocks, num_classes)


def _write_tensor_csv(tensor, csv_path):
    data = tensor.detach().cpu().numpy()
    if data.ndim == 1:
        flat_data = data.reshape(-1, 1)
    elif data.ndim > 2:
        flat_data = data.reshape(data.shape[0], -1)
    else:
        flat_data = data

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(flat_data)

    return data.shape


def export_ordered_weights_to_csv(model, output_dir, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    files_written = []

    for i, block in enumerate(model.blocks):
        dw_path = os.path.join(output_dir, f"blocks_{i}_dw_weight.csv")
        dw_shape = _write_tensor_csv(block.dw.weight, dw_path)
        files_written.append(dw_path)
        if verbose:
            print(f"Saved blocks.{i}.dw.weight -> {dw_path} (shape={dw_shape})")

        pw_path = os.path.join(output_dir, f"blocks_{i}_pw_weight.csv")
        pw_shape = _write_tensor_csv(block.pw.weight, pw_path)
        files_written.append(pw_path)
        if verbose:
            print(f"Saved blocks.{i}.pw.weight -> {pw_path} (shape={pw_shape})")

    fc_weight_path = os.path.join(output_dir, "classifier_weight.csv")
    fc_weight_shape = _write_tensor_csv(model.classifier.weight, fc_weight_path)
    files_written.append(fc_weight_path)
    if verbose:
        print(f"Saved classifier.weight -> {fc_weight_path} (shape={fc_weight_shape})")

    if model.classifier.bias is not None:
        fc_bias_path = os.path.join(output_dir, "classifier_bias.csv")
        fc_bias_shape = _write_tensor_csv(model.classifier.bias, fc_bias_path)
        files_written.append(fc_bias_path)
        if verbose:
            print(f"Saved classifier.bias -> {fc_bias_path} (shape={fc_bias_shape})")

    return files_written


def collect_ordered_csv_paths(csv_dir):
    csv_dir = Path(csv_dir)

    block_indices = sorted(
        {
            int(m.group(1))
            for path in csv_dir.glob("blocks_*_dw_weight.csv")
            for m in [re.match(r"blocks_(\d+)_dw_weight\.csv$", path.name)]
            if m
        }
    )

    ordered = []
    for i in block_indices:
        dw = csv_dir / f"blocks_{i}_dw_weight.csv"
        pw = csv_dir / f"blocks_{i}_pw_weight.csv"
        if dw.exists():
            ordered.append(str(dw))
        if pw.exists():
            ordered.append(str(pw))

    fc_weight = csv_dir / "classifier_weight.csv"
    fc_bias = csv_dir / "classifier_bias.csv"
    if fc_weight.exists():
        ordered.append(str(fc_weight))
    if fc_bias.exists():
        ordered.append(str(fc_bias))

    return ordered


def _pad_matrix(matrix, pad_rows, pad_cols):
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    for row in matrix:
        row.extend(["0"] * (pad_cols - n_cols))
    for _ in range(pad_rows - n_rows):
        matrix.append(["0"] * pad_cols)
    return matrix


def _process_csv_file(csv_file, out_fh, report, clamp=True, verbose=True):
    with open(csv_file, newline="") as cf:
        rows = list(csv.reader(cf))
    if not rows:
        if verbose:
            print(f"Warning: {csv_file} is empty.")
        return

    n_rows = len(rows)
    n_cols = len(rows[0])
    file_name = Path(csv_file).name

    if "_dw_weight.csv" in file_name or file_name == "classifier_weight.csv":
        pad_rows = next_power_of_two(n_rows)
        pad_cols = n_cols
        pad_desc = f"rows {n_rows}->{pad_rows}"
    elif "_pw_weight.csv" in file_name:
        pad_rows = n_rows
        pad_cols = next_power_of_two(n_cols)
        pad_desc = f"cols {n_cols}->{pad_cols}"
    else:
        pad_rows, pad_cols = n_rows, n_cols
        pad_desc = "none"

    if verbose and pad_desc != "none":
        print(f"Padding {file_name}: {pad_desc}")

    matrix = _pad_matrix(rows, pad_rows, pad_cols)

    for r_idx, row in enumerate(matrix):
        for c_idx, item in enumerate(row):
            item_strip = item.strip()
            if item_strip == "":
                continue
            try:
                val, corrected, reason = to_int8_safe(item_strip, clamp=clamp)
            except Exception as err:
                report.append((file_name, r_idx, c_idx, item_strip, "PARSE_ERR", str(err)))
                continue

            out_fh.write(f"{(val & 0xFF):02X}\n")
            if corrected:
                report.append((file_name, r_idx, c_idx, item_strip, "CLAMPED", reason))


def convert_csv_files_to_hex(csv_files, out_hex, clamp=True, verbose=True):
    report = []
    with open(out_hex, "w") as out_f:
        for csv_file in csv_files:
            if verbose:
                print(f"Processing {Path(csv_file).name} ...")
            _process_csv_file(csv_file, out_f, report, clamp=clamp, verbose=verbose)

    print(f"\nWrote combined hex -> {out_hex}")
    if report and verbose:
        print("\nDiagnostics:")
        for fname, r, c, raw, kind, info in report:
            print(f" {kind}: {fname} [row {r}, col {c}] raw='{raw}' -> {info}")
    elif verbose:
        print("No parsing/clamping issues detected.")

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export model weights to ordered CSV files and combine them into a single hex file."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--csv-dir", default=DEFAULT_CSV_DIR, help="Directory for CSV output")
    parser.add_argument(
        "--hex-out",
        default=DEFAULT_HEX_OUT,
        help="Output hex path",
    )
    parser.add_argument(
        "--temporal-length",
        type=int,
        default=128,
        help="Temporal length used when inferring spec (metadata only)",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Fail on values outside int8 range instead of clamping",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce log output")
    return parser.parse_args()


def main():
    args = parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required to run this script. Activate the environment where torch is installed.")

    from models.model_generator import Int8ModelFromSpec
    verbose = not args.quiet

    loaded = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_extract_state_dict(loaded))

    spec = infer_spec_from_state_dict(state_dict, temporal_length=args.temporal_length)
    if verbose:
        print(f"Inferred spec: {spec}")

    model = Int8ModelFromSpec(spec)
    model.load_state_dict(state_dict)

    csv_files = export_ordered_weights_to_csv(model, output_dir=args.csv_dir, verbose=verbose)
    if not csv_files:
        raise RuntimeError("No CSV files were written.")

    out_hex = args.hex_out or DEFAULT_HEX_OUT
    convert_csv_files_to_hex(csv_files, out_hex, clamp=not args.no_clamp, verbose=verbose)


if __name__ == "__main__":
    main()
