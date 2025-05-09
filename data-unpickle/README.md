# data-unpickle

This sub-project handles the extraction and conversion of legacy pickled datasets
(originally serialized using older versions of `pandas`) into modern CSV format.

## Compatibility

The `.pkl` files were created using an older version of pandas that is incompatible with newer releases. To unpickle them, we use:

- `pandas==1.5.3`
- `numpy==1.26.0`

Ensure that these exact versions are installed in a virtual environment before running the script.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
