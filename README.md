# README — CNN + LSTM (PyTorch)

**Short description**  
This repository contains a PyTorch implementation of a CNN → LSTM pipeline for sequence modelling (classification / regression) converted from an existing Jupyter notebook. It includes a dataset wrapper, model, training & evaluation loops, and utilities to save/load the best model. The focus here is on running everything in PyTorch, installation / dependencies, and clear instructions for training and inference.

---

## Repository structure (files you should have)
- `CNN+LSTM (1).ipynb`  
  The original Jupyter notebook you uploaded (contains dataset exploration, preprocessing and the original TensorFlow/Keras code). Keep it for reference and reproducibility. Use it to map exact preprocessing steps and label conventions into the PyTorch script.

- `Cnn Lstm Pytorch` *(canvas code file / script)*  
  A ready-to-run PyTorch script implementing:
  - `NumpySequenceDataset` — wraps numpy arrays of shape `(N, C, L)` (N samples, C channels / features, L timesteps).
  - `CNN_LSTM` model — Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout -> Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout -> LSTM -> FC.
  - training loop (`train_one_epoch`), evaluation (`evaluate`), checkpoint save/load and example data placeholders.
  Replace the placeholder data-loading block with your real arrays or a Dataset class that loads from files.

- `best_cnn_lstm.pth` *(output after training — optional)*  
  Example filename used by the script to save the best model checkpoint. Not present initially; created after training.

- `README.md` *(this file)*

- `requirements.txt` *(suggested; see below)*

---

## Requirements & Dependencies

Minimum recommended Python environment:
- Python 3.8+ (3.9 / 3.10 recommended)

Python packages (put these in `requirements.txt`):

```
torch>=1.13            # or latest compatible stable PyTorch for your CUDA
torchvision>=0.14      # optional (not used directly but commonly useful)
numpy>=1.21
tqdm>=4.60
matplotlib>=3.4        # optional, for plotting/training curves if you add them
scikit-learn>=1.0      # optional, for metrics / splits
jupyterlab             # optional if you run the notebook
```

If you have CUDA available, install the matching `torch` wheel from PyTorch site (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`), or use `conda` to manage CUDA-enabled builds.

---

## Quick setup

1. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate.bat       # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
If you need a GPU build of PyTorch, follow official PyTorch install instructions and install the correct `torch`/`torchvision` for your CUDA version.

---

## How data should be shaped / prepared

The PyTorch code expects your input arrays arranged as NumPy arrays with shape:
- `X`: `(N, C, L)` — N samples, C input channels (features per timestep), L length (timesteps)
- `y`: `(N,)` for classification integer labels OR `(N, num_targets)` for regression/multi-output

Make sure to:
- Convert label encoding to integer labels `0..(num_classes-1)` for `CrossEntropyLoss`.
- Standardize/normalize input features the same way you did in the notebook. The model code does not include dataset normalization steps — do them before constructing the dataset.

If your notebook used a different ordering (e.g. `(N, L, C)`), transpose to `(N, C, L)` before passing to `NumpySequenceDataset`.

---

## How to run training (example)

Open the Python script and update the placeholder section where random arrays are created. Replace with actual `X` and `y` arrays or wrap your on-disk dataset into `NumpySequenceDataset` or a custom `torch.utils.data.Dataset`.

Run the script (if saved as `train.py` or using the canvas file):

```bash
python Cnn_Lstm_Pytorch.py
```

Key parameters you can edit at the top of the script:
- `input_channels` — number of features per timestep (C)
- `seq_len` — sequence length (L)
- `num_classes` — set to `1` for regression, `>1` for classification
- `batch_size`, `epochs`, `lr` — training hyperparameters
- `device` selection: the script automatically uses CUDA if available

What the script does:
- Splits dataset into training and validation (`val_frac = 0.2`)
- Trains using `Adam` optimizer and `ReduceLROnPlateau` scheduler
- Uses gradient clipping (`clip_grad=5.0`) in the example
- Saves the best checkpoint to `best_cnn_lstm.pth` (contains `model_state_dict` and some meta info)

---

## Example: training command + tips

If you want to use arguments instead of editing the file, you can wrap the training part in a CLI (not included in the canvas file by default). For quick runs, edit hyperparameters directly in the script.

Tips:
- If your dataset is large, implement a custom `Dataset` that reads data on the fly (e.g., from HDF5 / memory-mapped files).
- Use `num_workers` in `DataLoader` to speed data loading: `DataLoader(..., num_workers=4)` (increase as appropriate).
- Use `pin_memory=True` when training on GPU for small speed improvements: `DataLoader(..., pin_memory=True)`.

---

## How to perform inference / load a saved checkpoint

Example Python snippet to load `best_cnn_lstm.pth` and run inference on new data:

```python
import torch
import numpy as np
from your_module import CNN_LSTM  # import from script or copy class into a new file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model must be constructed with the same architecture params used during training
model = CNN_LSTM(input_channels=6, num_classes=10, lstm_hidden=128, lstm_layers=1, dropout=0.3)
model.to(device)

ckpt = torch.load('best_cnn_lstm.pth', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# X_new must be a numpy array shape (N, C, L)
X_new = np.load('X_new.npy').astype('float32')
x_tensor = torch.from_numpy(X_new).to(device)

with torch.no_grad():
    out = model(x_tensor)   # shape (N, num_classes) or (N, 1)
    if out.shape[-1] > 1:
        preds = torch.argmax(out, dim=1).cpu().numpy()
    else:
        preds = out.cpu().numpy()
```

Make sure `input_channels`, `seq_len`, `num_classes`, and other architecture params exactly match the training configuration.

---

## Matching behaviour with original TensorFlow notebook

If your goal is to reproduce the TensorFlow model outputs exactly:
- Make sure convolutional kernel sizes, padding, pooling sizes and the order of layers are identical.
- Confirm random seed and deterministic backend settings if you need exact reproducibility (PyTorch/CUDA non-determinism can make bitwise-equal results difficult).
- Match preprocessing (normalization, windowing, label encoding) exactly.
- If your TF model used batch dimension ordering `(N, L, C)` or used `TimeDistributed(Dense(...))`, adapt the PyTorch permutes to match.

---

## Common issues & troubleshooting

- **Shape mismatch on LSTM input** — After Conv1D+pooling the tensor is `(N, channels, L')`; the script permutes to `(N, L', features)` before feeding LSTM. If your LSTM expects different features, adjust `permute(0,2,1)`.
- **CUDA out of memory** — reduce `batch_size` or move to smaller model (`lstm_hidden` / fewer filters).
- **Incorrect label types** — `CrossEntropyLoss` expects integer class labels `LongTensor`. Make sure `y` is integer dtype for classification.
- **Different performance vs TF** — tune learning rate, weight initialization, or add `torch.manual_seed(seed)` for deterministic runs.

---

## Suggested `requirements.txt` (example)

```
numpy>=1.21
torch>=1.13
tqdm
scikit-learn
matplotlib
jupyterlab
```

(If you need GPU support, install `torch` with the appropriate CUDA support as described on the official PyTorch install page.)

---

## Optional next steps
- Convert the exact cells from `CNN+LSTM (1).ipynb` into a runnable PyTorch notebook (with preprocessing and exact hyperparameters).
- Add CLI argument parsing (argparse) to the script for hyperparameters and file paths.
- Add logging (TensorBoard) and training metrics plotting.
- Add unit tests for dataset shapes and an end-to-end test run on a small dataset.

---
