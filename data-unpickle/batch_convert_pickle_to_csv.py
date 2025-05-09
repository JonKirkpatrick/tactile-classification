# Compatible environment:
# pandas==1.5.3
# numpy==1.26.0
# These versions are required to unpickle legacy data due to dtype and encoding compatibility.

import os
import pickle
import pandas as pd

INPUT_ROOT = "."
OUTPUT_ROOT = "CSV_Converted"

def convert_pickle_to_csv(input_path, output_path):
    """Convert a pickled pandas DataFrame to CSV format.

    Args:
        input_path (str): Path to the input .pkl file.
        output_path (str): Path to write the converted .csv file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
        PermissionError: If file access is denied.
        TypeError: If the unpickled object is not a DataFrame.
        OSError: For other I/O-related errors.
    """
    try:
        with open(input_path, "rb") as f:
            data = pd.compat.pickle_compat.load(f, encoding="latin1")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Unpickled object is not a pandas DataFrame.")

        if data.index.name is None:
            data.index.name = "timestamp"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=True)
        print(f"{input_path} -> {output_path}")

    except (FileNotFoundError, PermissionError, pickle.UnpicklingError,
            TypeError, OSError) as e:
        print(f"Error converting {input_path}: {e}")


def batch_convert_directory(input_root, output_root):
    """Traverse the input directory and convert all .pkl files to .csv."""
    for dirpath, _, filenames in os.walk(input_root):
        for file in filenames:
            if file.endswith(".pkl"):
                rel_path = os.path.relpath(os.path.join(dirpath, file), input_root)
                csv_path = os.path.join(output_root, rel_path).replace(".pkl", ".csv")
                convert_pickle_to_csv(os.path.join(dirpath, file), csv_path)


if __name__ == "__main__":
    batch_convert_directory(INPUT_ROOT, OUTPUT_ROOT)
