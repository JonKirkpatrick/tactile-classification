import numpy as np

data = np.load("data/int8_dataset.npz")

for k in data.files:
    arr = data[k]
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}, order={'C' if arr.flags['C_CONTIGUOUS'] else 'F'}")

