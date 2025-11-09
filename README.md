# Anomaly Detection in Network Traffic (PyTorch)

## 🧠 Project Overview
This project implements a **PyTorch-based CNN + LSTM model** for anomaly detection in network traffic, adapted and extended from the repository [Snehasawla/Anomaly-Detection-in-Network-Traffic](https://github.com/Snehasawla/Anomaly-Detection-in-Network-Traffic).  
The objective is to automatically detect abnormal or malicious network behavior using deep sequence modeling.

---

## 📂 Repository Structure

| File | Description |
|------|--------------|
| `CNN+LSTM (1).ipynb` | Original notebook for preprocessing, dataset exploration, and baseline model (TensorFlow/Keras). |
|`DDosNotebook (2).ipynb`| Exploratory data analysis (EDA) of network traffic data (e.g., CICIDS, UNSW-NB15, or custom traffic logs).|
| `Cnn_Lstm_Pytorch.py` | PyTorch reimplementation of CNN + LSTM pipeline. Includes model definition, training, validation, and inference. |
| `best_cnn_lstm.pth` | Saved checkpoint file for best model (automatically generated after training). |
| `requirements.txt` | List of Python dependencies required for this project. |
| `README.md` | Documentation for installation, usage, and understanding the project. |

---

## ⚙️ Requirements & Setup

### **1️⃣ Environment Setup**
It is recommended to create a virtual environment before installation.

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate.bat       # Windows
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

If GPU support is available, install the CUDA-compatible version of PyTorch from [PyTorch.org](https://pytorch.org/get-started/locally/).

### **Example requirements.txt**
```
numpy>=1.21
torch>=1.13
tqdm
scikit-learn
matplotlib
jupyterlab
ipykernel
```

---

## 🧩 Model Architecture

The CNN + LSTM architecture captures both **spatial patterns** (via CNN) and **temporal dependencies** (via LSTM).

```
Input (N, C, L)
   │
   ├── Conv1D(128 filters, kernel=3) + BatchNorm + ReLU + MaxPool(2)
   ├── Conv1D(256 filters, kernel=3) + BatchNorm + ReLU + MaxPool(2)
   ├── Dropout(0.3)
   ├── LSTM(hidden_size=128)
   ├── Fully Connected Layer → num_classes
Output: Classification (normal / anomaly)
```

Where:
- `N`: Number of samples  
- `C`: Input features per timestep  
- `L`: Sequence length

---

## 📊 Data Preparation

The model expects **NumPy arrays** as input.

| Array | Shape | Description |
|--------|--------|-------------|
| `X` | `(N, C, L)` | Input sequence data (C = features, L = timesteps) |
| `y` | `(N,)` or `(N, num_classes)` | Labels (integer encoded for classification) |

**Important preprocessing notes:**
- Normalize / standardize feature values as done in the original notebook.
- Transpose arrays from `(N, L, C)` → `(N, C, L)` if necessary.
- Ensure integer labels for classification tasks.

---

## 🚀 Training Instructions

1. Update hyperparameters inside `Cnn_Lstm_Pytorch.py`:
   ```python
   input_channels = 6     # Number of input features
   seq_len = 128          # Sequence length
   num_classes = 2        # Binary classification (Normal, Anomaly)
   batch_size = 64
   epochs = 20
   lr = 1e-3
   ```

2. Replace the placeholder dataset generation block with your actual data loading logic (e.g., from CSV or `.npy` files).

3. Start training:
   ```bash
   python Cnn_Lstm_Pytorch.py
   ```

4. The model automatically:
   - Splits the dataset into train/validation sets.
   - Trains using **Adam optimizer** and **ReduceLROnPlateau** scheduler.
   - Applies **gradient clipping (clip_grad=5.0)**.
   - Saves the best checkpoint to `best_cnn_lstm.pth`.

---

## 🧠 Inference Instructions

To make predictions using a trained model:

```python
import torch
import numpy as np
from Cnn_Lstm_Pytorch import CNN_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN_LSTM(input_channels=6, num_classes=2, lstm_hidden=128, lstm_layers=1, dropout=0.3)
model.load_state_dict(torch.load('best_cnn_lstm.pth', map_location=device)['model_state_dict'])
model.eval()

X_new = np.load('new_network_data.npy').astype('float32')
x_tensor = torch.from_numpy(X_new).to(device)

with torch.no_grad():
    outputs = model(x_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

print("Predicted labels:", preds)
```

Ensure all model parameters (`input_channels`, `seq_len`, `num_classes`) match those used during training.

---

## 🔍 Relation to Original Project

This PyTorch version aligns with the goals of the [original project](https://github.com/Snehasawla/Anomaly-Detection-in-Network-Traffic), which aims to detect anomalies in network traffic using machine learning.

Enhancements in this implementation:
- Rewritten using **PyTorch** instead of TensorFlow/Keras.
- Modularized training and evaluation loops.
- Reproducible dataset handling (`NumpySequenceDataset`).
- Model checkpointing for best validation performance.
- Compatible with GPU acceleration.

---

## 🧩 Future Improvements
- Integrate **Bidirectional LSTM** for better temporal representation.
- Add **attention mechanism** post-LSTM for weighted feature extraction.
- Implement **real-time anomaly detection** using streaming data.
- Add **evaluation metrics dashboard** (Precision, Recall, ROC-AUC).
- Include **hyperparameter optimization** (Optuna / Ray Tune).

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|-----------|
| Shape mismatch | Verify input shape `(N, C, L)` before feeding into the model. |
| CUDA out of memory | Reduce batch size or model size. |
| Incorrect label dtype | Convert labels to `torch.LongTensor`. |
| Model not converging | Check learning rate and normalization of input data. |

---

## 📜 License & Credits

This work builds upon [Snehasawla/Anomaly-Detection-in-Network-Traffic](https://github.com/Snehasawla/Anomaly-Detection-in-Network-Traffic).  
The PyTorch implementation and this README were developed for academic and research purposes.  
Please review the original repository’s license before reuse or distribution.

---

**Maintainer:** Sneha Sawla  
**Framework:** PyTorch 2.x  
**Purpose:** Network Traffic Anomaly Detection using CNN + LSTM  
**Year:** 2025
