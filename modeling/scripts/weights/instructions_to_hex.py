import json
import math
import argparse
from pathlib import Path
from typing import Dict, Iterable

try:
    from _bootstrap import ensure_modeling_root_on_path
except ImportError:
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from _bootstrap import ensure_modeling_root_on_path

MODEL_ROOT = ensure_modeling_root_on_path()

# ----------------------
# Small utilities
# ----------------------
def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())

def log2_pow2(n: int) -> int:
    """Return log2(n) assuming n is exact power of two."""
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1

def ceil_log2_for_value(n: int) -> int:
    """Return the smallest k such that 2**k >= n (k>=0)."""
    if n <= 1:
        return 0
    return (n - 1).bit_length()

# bit packing helper (lsb index convention)
def set_bits(word: int, value: int, lsb: int, width: int) -> int:
    if width <= 0:
        return word
    mask = ((1 << width) - 1) << lsb
    value_masked = (value & ((1 << width) - 1)) << lsb
    word = (word & ~mask) | value_masked
    return word & 0xFFFFFFFF

def write_hex_file(words: Iterable[int], path: str):
    with open(path, "w") as f:
        for w in words:
            f.write(f"{w:08X}\n")

# ----------------------
# Field positions per PDF (LSB index, width)
# ----------------------
BITFIELD = {
    "pkt_size":   (0,  4),   # bits 3:0
    "shift_amt":  (4,  4),   # bits 7:4
    "time_steps": (8,  4),   # bits 11:8
    "padding":    (12, 4),   # bits 15:12
    "out_ch":     (16, 4),   # bits 19:16 (log2)
    "in_ch":      (20, 4),   # bits 23:20 (log2)
    "maxpool":    (24, 4),   # bits 27:24 (log2)
    "global_avg": (28, 1),   # bit 28
    # bits 29..31 unused per PDF (keep zero)
}

# ----------------------
# Main encoding: map an IR layer -> single 32-bit control word
# ----------------------
def encode_layer_word(layer: Dict) -> int:
    """
    Returns a 32-bit integer representing the instruction for `layer`.
    Encoding rules follow the PDF:
      - channel fields are encoded as log2(padded_channel_count)
      - pkt_size is ceil_log2 of the relevant dot product length
      - time_steps uses the current temporal length
    """
    w = 0

    ltype = layer["type"]

    # Prepare common fields
    # For time_steps: input temporal dimension BEFORE this layer.
    # IR stores temporal influence only in pooling; but we included collapse_factor in GAP.
    # For our IR, we compute time_steps as follows:
    #   - For non-gap layers: time_steps = the current temporal length when layer appears.
    #     We cannot necessarily recover that from a single layer entry, so the caller
    #     should supply a 'time_steps_before' field or we can derive by simulating through IR.
    # To keep this function pure, we assume the layer dict includes "time_steps" (absolute value).
    # If absent, default to 1 (log2 = 0).
    time_steps = layer.get("time_steps", 1)
    if time_steps <= 0:
        raise ValueError("time_steps must be >= 1")

    # Shift amount
    shift_log2 = layer.get("shift_log2", 0) or 0

    # Packet size calculation depends on layer type
    if ltype == "depthwise_conv1d":
        kernel = layer.get("kernel", 1)
        pkt_k = kernel
        # padding is literal padding per side
        padding = layer.get("padding", 0) or 0
        # in/out channels are same, pad them
        in_ch = layer["in_channels"]
        out_ch = layer["out_channels"]
        in_hw = next_pow2(in_ch)
        out_hw = next_pow2(out_ch)

    elif ltype == "pointwise_conv1d":
        # pointwise uses the channel dimension as dot product length
        in_ch = layer["in_channels"]
        out_ch = layer["out_channels"]
        in_hw = next_pow2(in_ch)
        out_hw = next_pow2(out_ch)
        pkt_k = in_hw  # dot products of length equal to padded in_channels
        padding = 0

    elif ltype == "fully_connected":
        # FC: dot product length equals in_features (channels after GAP)
        in_f = layer["in_features"]
        out_f = layer["out_features"]
        in_hw = next_pow2(in_f)
        out_hw = next_pow2(out_f)
        pkt_k = in_hw
        padding = 0

    elif ltype == "global_avg_pool":
        # Global avg has little else; encode as a special instruction with global_avg bit set.
        collapse = layer.get("collapse_factor", 1)
        if collapse <= 0:
            raise ValueError("collapse_factor must be >= 1")
        in_ch = layer["in_channels"]
        out_ch = layer["out_channels"]
        in_hw = next_pow2(in_ch)
        out_hw = next_pow2(out_ch)
        pkt_k = collapse
        padding = 0

    else:
        raise KeyError(f"Unsupported layer type for encoding: {ltype}")

    # Compute log2 fields
    in_ch_log2 = log2_pow2(in_hw)
    out_ch_log2 = log2_pow2(out_hw)
    time_steps_log2 = ceil_log2_for_value(time_steps)  # smallest k such that 2**k >= time_steps
    # But PDF expects Time Steps = log2(T) where T is input temporal dimension exact power of two.
    # If time_steps is not exact power-of-two, we still encode ceil_log2 so hardware sees the size.
    # If you want strict checking, replace with log2_pow2(time_steps).
    # For now we enforce power-of-two: prefer to error if not power of two:
    if (time_steps & (time_steps - 1)) != 0:
        # not power-of-two: warn but still encode ceil_log2 (hardware uses power-of-two assumption)
        # We'll keep the ceil value.
        pass

    pkt_log2 = ceil_log2_for_value(int(pkt_k))
    # MaxPool (log2) if present on pointwise layer else 0
    pool_factor = layer.get("pool_factor", None) or 0
    if ltype == "global_avg_pool":
        pool_factor = layer.get("collapse_factor", 1)
    maxpool_log2 = 0
    if pool_factor:
        maxpool_log2 = ceil_log2_for_value(int(pool_factor))

    # Build word piecewise using BITFIELD mapping
    w = set_bits(w, pkt_log2, *BITFIELD["pkt_size"])
    w = set_bits(w, shift_log2, *BITFIELD["shift_amt"])
    w = set_bits(w, time_steps_log2, *BITFIELD["time_steps"])
    w = set_bits(w, int(padding), *BITFIELD["padding"])
    w = set_bits(w, out_ch_log2, *BITFIELD["out_ch"])
    w = set_bits(w, in_ch_log2, *BITFIELD["in_ch"])
    w = set_bits(w, maxpool_log2, *BITFIELD["maxpool"])
    # global avg bit:
    if ltype == "global_avg_pool":
        w = set_bits(w, 1, *BITFIELD["global_avg"])
    else:
        w = set_bits(w, 0, *BITFIELD["global_avg"])

    return w

# ----------------------
# Top-level: walk IR and emit words
# ----------------------
def ir_to_words(ir: Dict) -> Iterable[int]:
    """
    Walk the IR and generate a sequence of 32-bit control words.
    This function will simulate the temporal length across layers so that 'time_steps'
    passed to encode_layer_word is correct.
    """
    words = []
    temporal = ir["model"]["input"]["temporal_length"]
    for layer in ir["model"]["layers"]:
        ltype = layer["type"]

        # Provide the 'time_steps' to the encoder so TimeSteps is correct
        layer_local = dict(layer)  # shallow copy
        layer_local["time_steps"] = temporal

        # Encode the layer into a single word
        word = encode_layer_word(layer_local)
        words.append(word)

        # Update temporal for next layer: if this layer had pooling, divide by pool_factor
        pool = layer.get("pool_factor", None)
        if pool:
            temporal = max(1, temporal // int(pool))

        # If GAP or final FC, temporal becomes 1 (GAP reduces to 1)
        if ltype == "global_avg_pool":
            temporal = 1

    # Hardware protocol terminator:
    # 0x00000000 => ArgMax trigger
    # next word   => number of classes
    num_classes = ir["model"]["output"]["target_count"]
    words.append(0x00000000)
    words.append(int(num_classes) & 0xFFFFFFFF)

    return words

# ----------------------
# CLI-ish usage
# ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert model IR JSON into ordered instruction hex")
    parser.add_argument("--ir", default=str(MODEL_ROOT / "artifacts" / "weights" / "ir" / "model_ir.json"), help="Path to the model IR JSON file")
    parser.add_argument("--out", default=str(MODEL_ROOT / "artifacts" / "weights" / "hex" / "model_instructions.hex"), help="Output hex file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise SystemExit(f"Place your IR JSON at {ir_path} (output from spec parser)")

    ir = json.loads(ir_path.read_text())

    words = list(ir_to_words(ir))

    write_hex_file(words, args.out)
    print(f"Wrote {len(words)} words to {args.out}")
