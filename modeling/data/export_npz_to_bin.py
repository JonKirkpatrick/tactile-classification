import numpy as np
import struct
import sys
from pathlib import Path

MAGIC = 0x44544E49  # 'INTD'
VERSION = 1
LOGICAL_CHANNELS = 13
PHYSICAL_CHANNELS = 16
TIMESTEPS = 128

def export_split(npz_path, x_key, y_key, out_path):
    data = np.load(npz_path)

    X = data[x_key]
    y = data[y_key]

    # ----------------------------
    # Assertions (do not soften)
    # ----------------------------
    assert X.dtype == np.int8
    assert y.dtype == np.int32
    assert X.ndim == 3
    assert X.shape[1] == TIMESTEPS
    assert X.shape[2] == LOGICAL_CHANNELS
    assert np.all((y >= 0) & (y < 256))

    N = X.shape[0]

    # ----------------------------
    # Transpose: (N, 128, 13) -> (N, 13, 128)
    # ----------------------------
    X = np.transpose(X, (0, 2, 1))

    # ----------------------------
    # Pad channels to power-of-two
    # ----------------------------
    X_pad = np.zeros((N, PHYSICAL_CHANNELS, TIMESTEPS), dtype=np.int8)
    X_pad[:, :LOGICAL_CHANNELS, :] = X

    labels = y.astype(np.uint8)

    # ----------------------------
    # Write binary
    # ----------------------------
    with open(out_path, "wb") as f:
        f.write(struct.pack(
            "<6I",
            MAGIC,
            VERSION,
            N,
            LOGICAL_CHANNELS,
            PHYSICAL_CHANNELS,
            TIMESTEPS
        ))
        f.write(X_pad.tobytes(order="C"))
        f.write(labels.tobytes(order="C"))

    print(f"Wrote {out_path} ({N} samples)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: export_npz_to_bin.py <dataset.npz> <out_dir>")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    export_split(npz_path, "X_train", "y_train", out_dir / "train.bin")
    export_split(npz_path, "X_val",   "y_val",   out_dir / "val.bin")
    export_split(npz_path, "X_test",  "y_test",  out_dir / "test.bin")
