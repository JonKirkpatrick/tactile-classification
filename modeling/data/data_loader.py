import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy.ndimage import uniform_filter1d

def apply_temporal_smoothing(X, window_size=13):
    """
    Applies causal moving average smoothing over the last axis (time).
    """
    return uniform_filter1d(X, size=window_size, axis=-1, mode='nearest')

def load_data_sample(dataset_path, sample_size=1, batch_size=256, smooth=False, window_size=13):
    """
    Loads and returns sampled DataLoaders for training and validation.

    Parameters:
        dataset_path (str): Path to .npz dataset file.
        sample_size (float): Fraction of data to sample from each split, or 0 to use all data.
        batch_size (int): Batch size for DataLoaders.
        smooth (bool): Whether to apply causal moving average smoothing.
        window_size (int): Window size for smoothing.

    Returns:
        (train_loader, val_loader): Tuple of DataLoader objects.
    """
    data = np.load(dataset_path)
    X_train_full, y_train_full = data['X_train'], data['y_train']
    X_val_full, y_val_full = data['X_val'], data['y_val']

    if 0 < sample_size < 1:
        _, X_train, _, y_train = train_test_split(X_train_full, y_train_full, test_size=sample_size, stratify=y_train_full)
        _, X_val, _, y_val = train_test_split(X_val_full, y_val_full, test_size=sample_size, stratify=y_val_full)
    else:
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = X_val_full, y_val_full

    if smooth:
        X_train = apply_temporal_smoothing(X_train, window_size=window_size)
        X_val = apply_temporal_smoothing(X_val, window_size=window_size)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

    return train_loader, val_loader
