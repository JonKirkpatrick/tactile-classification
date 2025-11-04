import os
import numpy as np

# Configuration
INPUT_DIR = "./sequences_by_texture"
OUTPUT_PATH = "./balanced_dataset.npz"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15  # Test = remaining 0.15
SEED = 42
np.random.seed(SEED)

X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for label in range(12):
    class_path = os.path.join(INPUT_DIR, f"sequences_texture_{label+1:02d}.npz")
    data = np.load(class_path)
    X = data['X']
    y = data['y']

    assert np.all(y == label), f"Mismatched label in {class_path}"

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train.append(X[train_idx])
    y_train.append(y[train_idx])
    X_val.append(X[val_idx])
    y_val.append(y[val_idx])
    X_test.append(X[test_idx])
    y_test.append(y[test_idx])

# Concatenate and shuffle each set
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]

X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)
perm = np.random.permutation(len(X_val))
X_val, y_val = X_val[perm], y_val[perm]

X_test = np.concatenate(X_test)
y_test = np.concatenate(y_test)
perm = np.random.permutation(len(X_test))
X_test, y_test = X_test[perm], y_test[perm]

np.savez_compressed(
    OUTPUT_PATH,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test
)

print(f"✓ Balanced dataset saved to {OUTPUT_PATH}")
print(f"  → Train: {len(X_train)} samples")
print(f"  → Val:   {len(X_val)} samples")
print(f"  → Test:  {len(X_test)} samples")
