import torch
import numpy as np
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

def summarize_int8_state_dict(model_path):
    # Load the state_dict
    state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"{'Layer Name':<30} | {'Mean':<8} | {'Median':<8} | {'Std':<8} | {'Unique Vals'}")
    print("-" * 80)
    
    for name, param in state_dict.items():
        # Only analyze weights and biases (skip internal buffers if any)
        if isinstance(param, torch.Tensor):
            flattened = param.detach().cpu().numpy().flatten()
            
            # Calculate stats
            p_mean = np.mean(flattened)
            p_median = np.median(flattened)
            p_std = np.std(flattened)
            
            # Get a sample of unique values to verify integer-ness
            unique_samples = np.unique(flattened)
            if len(unique_samples) > 10:
                unique_str = f"{list(unique_samples[:5])} ... {list(unique_samples[-5:])}"
            else:
                unique_str = str(list(unique_samples))

            print(f"{name:<30} | {p_mean:>8.2f} | {p_median:>8.2f} | {p_std:>8.2f} | {unique_str}")

            # Optional: Print a small 5x5 raw sample for the first few weights
            if "weight" in name and "blocks" in name:
                sample = flattened[:5]
                print(f"   --> Raw Sample: {sample}")

if __name__ == "__main__":
    summarize_int8_state_dict(str(MODEL_ROOT / "artifacts" / "models" / "int8_model.pt"))