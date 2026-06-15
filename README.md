# Corn Hyperspectral Classification

This repository contains an end-to-end Python project for hyperspectral-based corn seed classification.

Contents
- src/: core code (dataset, models, training, utilities)
- requirements.txt: Python dependencies

Quickstart
1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # on Windows use venv\Scripts\activate
pip install -r requirements.txt
```

2. Prepare your data:
- CSV format: each row a sample, columns: band_1,...,band_L,label
- MAT format: a .mat file with keys `spectra` (N,B) and `labels` (N,)
Place your data under `data/` and provide the path to the training script.

3. Run training (example - RandomForest):

```bash
python src/train.py --data_path data/corn_spectra.csv --format csv --mode sklearn --clf rf
```

4. Run training (example - 1D-CNN):

```bash
python src/train.py --data_path data/corn_spectra.csv --format csv --mode torch --epochs 80 --batch_size 64
```

License: MIT
