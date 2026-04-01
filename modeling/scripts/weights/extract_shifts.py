#!/usr/bin/env python3
import argparse
import json
import math
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

import torch

from data.data_loader import load_data_sample
from models.model_generator import Int8ModelFromSpec
from utils.model_utils import ShiftClamp


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


def _shift_from_input(x):
    max_val = x.detach().abs().max()
    if max_val.item() == 0.0:
        shift_exp = 0.0
        scale = 1.0
    else:
        shift_exp = torch.ceil(torch.log2(max_val / 127.0)).item()
        scale = torch.pow(torch.tensor(2.0, device=x.device), torch.tensor(shift_exp, device=x.device)).item()
    return float(shift_exp), float(scale)


def extract_layer_shifts(model, data_loader, device, output_path, max_batches=1):
    model = model.to(device)
    model.eval()

    stats = {}

    def make_hook(name):
        def hook(module, inputs, outputs):
            x = inputs[0]
            shift_exp, scale = _shift_from_input(x)
            if name not in stats:
                stats[name] = {"shift_exp": shift_exp, "scale": scale, "seen": 1}
            else:
                # Conservative export: keep the largest shift seen across sampled batches.
                if shift_exp > stats[name]["shift_exp"]:
                    stats[name]["shift_exp"] = shift_exp
                    stats[name]["scale"] = scale
                stats[name]["seen"] += 1
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, ShiftClamp):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            _ = model(inputs.to(device))

    for hook in hooks:
        hook.remove()

    serialized = {
        name: {"shift_exp": values["shift_exp"], "scale": values["scale"]}
        for name, values in stats.items()
    }
    Path(output_path).write_text(json.dumps(serialized, indent=2))
    print(f"Saved {len(serialized)} layer shifts to {Path(output_path).resolve()}")
    return serialized


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-layer ShiftClamp metadata from checkpoint + sample data")
    parser.add_argument("--checkpoint", default=str(MODEL_ROOT / "artifacts" / "models" / "int8_model.pt"), help="Path to model checkpoint")
    parser.add_argument("--dataset-path", default=str(MODEL_ROOT / "data" / "int8_dataset.npz"), help="Path to dataset npz")
    parser.add_argument("--output", default=str(MODEL_ROOT / "artifacts" / "weights" / "metadata" / "layer_shifts.json"), help="Output shift metadata JSON")
    parser.add_argument("--temporal-length", type=int, default=128)
    parser.add_argument("--sample-size", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=13)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    loaded = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _normalize_state_dict_keys(_extract_state_dict(loaded))
    spec = infer_spec_from_state_dict(state_dict, temporal_length=args.temporal_length)
    print(f"Inferred spec: {spec}")

    model = Int8ModelFromSpec(spec)
    model.load_state_dict(state_dict)

    train_loader, _ = load_data_sample(
        args.dataset_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        smooth=args.smooth,
        window_size=args.smooth_window,
    )

    extract_layer_shifts(
        model,
        train_loader,
        device=device,
        output_path=args.output,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()