import json
import re
import argparse
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


def load_shift_map(shift_json):
    """
    Converts your PyTorch-exported training-time shift JSON into a flat map:
        IR-layer-name  -> shift_log2
    Supports legacy and current naming conventions, for example:
        blocks.N.block.1        -> blkN_depthwise
        blocks.N.block.3        -> blkN_pointwise
        blocks.N.dw / sc_dw     -> blkN_depthwise
        blocks.N.pw / sc_pw     -> blkN_pointwise
        classifier(.k)?         -> final_fc
    """
    shift_map = {}

    def read_shift_exp(entry):
        if isinstance(entry, dict):
            for key in ("shift_exp", "shift", "value"):
                if key in entry:
                    return int(entry[key])
            raise KeyError(f"No supported shift field in entry: {entry}")
        return int(entry)

    for key, entry in shift_json.items():
        shift_exp = read_shift_exp(entry)

        # Legacy pattern: blocks.N.block.K
        m = re.match(r"blocks\.(\d+)\.block\.(\d+)", key)
        if m:
            blk = int(m.group(1))
            sub = int(m.group(2))

            if sub == 1:
                name = f"blk{blk}_depthwise"
            elif sub == 3:
                name = f"blk{blk}_pointwise"
            else:
                continue

            shift_map[name] = shift_exp
            continue

        # Current patterns: blocks.N.dw / blocks.N.sc_dw
        m = re.match(r"blocks\.(\d+)\.(dw|sc_dw)$", key)
        if m:
            blk = int(m.group(1))
            shift_map[f"blk{blk}_depthwise"] = shift_exp
            continue

        # Current patterns: blocks.N.pw / blocks.N.sc_pw
        m = re.match(r"blocks\.(\d+)\.(pw|sc_pw)$", key)
        if m:
            blk = int(m.group(1))
            shift_map[f"blk{blk}_pointwise"] = shift_exp
            continue

        # Classifier patterns: classifier or classifier.k
        m = re.match(r"classifier(?:\.(\d+))?$", key)
        if m:
            shift_map["final_fc"] = shift_exp
            continue

        # Current model naming: final_shift module before classifier.
        if key == "final_shift":
            shift_map["final_fc"] = shift_exp
            continue

    return shift_map


def _extract_state_dict(loaded):
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        return loaded["state_dict"]
    return loaded


def _normalize_state_dict_keys(state_dict):
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


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
        raise ValueError("Could not infer block structure from state_dict keys")

    blocks = []
    for i in block_ids:
        dw_key = f"blocks.{i}.dw.weight"
        pw_key = f"blocks.{i}.pw.weight"
        if dw_key not in state_dict or pw_key not in state_dict:
            raise KeyError(f"Missing expected keys: {dw_key} and/or {pw_key}")

        dw_weight = state_dict[dw_key]
        pw_weight = state_dict[pw_key]
        kernel = int(dw_weight.shape[2])
        out_channels = int(pw_weight.shape[0])
        blocks.append((kernel, out_channels, None))

    sensor_channels = int(state_dict[f"blocks.{block_ids[0]}.dw.weight"].shape[0])
    if "classifier.weight" not in state_dict:
        raise KeyError("Missing classifier.weight in checkpoint")
    num_classes = int(state_dict["classifier.weight"].shape[0])

    return (sensor_channels, temporal_length, blocks, num_classes)


def build_ir_from_spec(spec, shift_json):
    """
    Build the IR dict from the (sensor_channels, temporal, blocks, target_count)
    and the training-time shift JSON.
    """
    sensor_channels, temporal_length, blocks, target_count = spec

    layer_shifts = load_shift_map(shift_json)

    current_channels = sensor_channels
    layers = []

    # ------------------------------------------
    # Blocks: (kernel, out_channels, pool_factor)
    # ------------------------------------------
    for blk_idx, (kernel, pw_out_channels, pool_factor) in enumerate(blocks):

        # Depthwise
        depth_name = f"blk{blk_idx}_depthwise"
        layers.append({
            "name": depth_name,
            "type": "depthwise_conv1d",
            "kernel": kernel,
            "stride": 1,
            "padding": (kernel - 1) // 2,
            "in_channels": current_channels,
            "out_channels": current_channels,
            "activation": None,
            "shift_log2": layer_shifts.get(depth_name)
        })

        # Pointwise
        pw_name = f"blk{blk_idx}_pointwise"
        layers.append({
            "name": pw_name,
            "type": "pointwise_conv1d",
            "kernel": 1,
            "stride": 1,
            "padding": 0,
            "in_channels": current_channels,
            "out_channels": pw_out_channels,
            "pool_factor": pool_factor,
            "activation": "relu",
            "shift_log2": layer_shifts.get(pw_name)
        })

        # Update tensor shape
        current_channels = pw_out_channels
        if pool_factor:
            temporal_length //= pool_factor

    # ------------------------------------------
    # Global Average Pooling
    # ------------------------------------------
    layers.append({
        "name": "gap",
        "type": "global_avg_pool",
        "in_channels": current_channels,
        "out_channels": current_channels,
        "collapse_factor": temporal_length,
    })

    # GAP collapses temporal dim:
    temporal_length = 1

    # ------------------------------------------
    # Fully Connected Layer
    # ------------------------------------------
    layers.append({
        "name": "final_fc",
        "type": "fully_connected",
        "in_features": current_channels,
        "out_features": target_count,
        "activation": None,
        "shift_log2": layer_shifts.get("final_fc")
    })

    # ------------------------------------------
    # Final IR
    # ------------------------------------------
    ir = {
        "model": {
            "input": {
                "channels": sensor_channels,
                "temporal_length": spec[1]
            },
            "layers": layers,
            "output": {
                "final_channels": current_channels,
                "final_temporal_length": 1,
                "target_count": target_count
            }
        }
    }

    return ir


# ------------------------------------------
# CLI entrypoint — produce model_ir.json
# ------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build model_ir.json from model spec and shift metadata")
    parser.add_argument("--shift-json", default=str(MODEL_ROOT / "artifacts" / "weights" / "metadata" / "layer_shifts.json"), help="Path to shift metadata JSON")
    parser.add_argument("--out", default=str(MODEL_ROOT / "artifacts" / "weights" / "ir" / "model_ir.json"), help="Output IR JSON path")
    parser.add_argument(
        "--checkpoint",
        default=str(MODEL_ROOT / "artifacts" / "models" / "int8_model.pt"),
        help="Optional checkpoint to infer spec automatically from state_dict",
    )
    parser.add_argument("--temporal-length", type=int, default=128)
    args = parser.parse_args()

    # Default fallback spec for compatibility when checkpoint is not supplied.
    spec = (
        13,
        args.temporal_length,
        [(7, 64, None), (3, 128, 2), (13, 256, 16), (3, 256, None)],
        12,
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path and checkpoint_path.exists():
        try:
            import torch
        except ModuleNotFoundError:
            raise SystemExit("PyTorch is required to infer spec from checkpoint.")

        loaded = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _normalize_state_dict_keys(_extract_state_dict(loaded))
        spec = infer_spec_from_state_dict(state_dict, temporal_length=args.temporal_length)
        print(f"Inferred spec from checkpoint: {spec}")
    else:
        if args.checkpoint:
            print(f"Checkpoint not found at {args.checkpoint}; using fallback spec.")

    shift_json_path = Path(args.shift_json)
    if not shift_json_path.exists():
        raise SystemExit(f"Shift JSON not found: {shift_json_path}")

    shift_json = json.loads(shift_json_path.read_text())
    ir = build_ir_from_spec(spec, shift_json)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(ir, indent=2))
    print(f"Wrote IR to {out_path.resolve()}")
