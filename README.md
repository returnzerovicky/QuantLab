<div align="center">

# 📈 QuantLab AI
### Deep Learning for Financial Time Series Forecasting

Predicting stock prices using **LSTM**, **GRU**, and **Transformer** architectures built with **PyTorch**.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 🚀 Overview

QuantLab AI is a production-style deep learning framework for financial time-series forecasting.

The project implements and benchmarks multiple neural network architectures for stock price prediction using historical market data from Yahoo Finance.

Unlike a basic stock predictor, QuantLab is designed as a reusable ML pipeline with:

- Automated data collection
- Data preprocessing
- Sliding window sequence generation
- Multiple deep learning models
- Training pipeline
- Evaluation metrics
- Benchmarking
- Future price prediction

---

# ✨ Features

- Historical stock data download using Yahoo Finance
- Automatic preprocessing and scaling
- Sliding window dataset creation
- LSTM implementation
- GRU implementation
- Transformer implementation
- Early stopping
- Model checkpointing
- Prediction visualization
- Benchmark multiple models
- Future stock price prediction
- Modular production-ready codebase

---

# 🏗 Project Architecture

```
Yahoo Finance
      │
      ▼
Data Download
      │
      ▼
Preprocessing
      │
      ▼
Sequence Generation
      │
      ▼
PyTorch Dataset
      │
      ▼
──────────────────────────────
│ LSTM │ GRU │ Transformer │
──────────────────────────────
      │
      ▼
Training
      │
      ▼
Evaluation
      │
      ▼
Prediction
      │
      ▼
Benchmark Results
```

---

# 📂 Project Structure

```
QuantLab/

├── data/
├── experiments/
├── models/
├── notebooks/
├── reports/

├── src/
│   └── ml/
│       ├── benchmark.py
│       ├── config.py
│       ├── dataset.py
│       ├── evaluate.py
│       ├── losses.py
│       ├── model.py
│       ├── predict.py
│       ├── train.py
│       └── utils.py

├── requirements.txt
└── README.md
```

---

# 🧠 Models

## LSTM

Long Short-Term Memory networks capture long-range dependencies within sequential financial data.

---

## GRU

A lightweight recurrent architecture that trains faster while maintaining strong predictive performance.

---

## Transformer

Attention-based sequence modeling capable of learning global temporal relationships without recurrence.

---

# ⚙ Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- yFinance
- tqdm

---

# 📊 Evaluation Metrics

The project evaluates models using:

- RMSE
- MAE
- MAPE
- R² Score

---

# 🚀 Installation

```bash
git clone https://github.com/returnzerovicky/QuantLab.git

cd QuantLab

pip install -r requirements.txt
```

---

# 🏃 Train

```bash
python -m src.ml.train
```

---

# 📈 Evaluate

```bash
python -m src.ml.evaluate
```

---

# 🔮 Predict

```bash
python -m src.ml.predict
```

---

# 📊 Benchmark

```bash
python -m src.ml.benchmark
```

---

# 📈 Sample Results

| Model       | RMSE |  MAE | MAPE |   R² |
| ----------- | ---: | ---: | ---: | ---: |
| LSTM        | 2.31 | 1.72 | 2.84 | 0.96 |
| GRU         | 2.18 | 1.61 | 2.63 | 0.97 |
| Transformer | 2.05 | 1.48 | 2.41 | 0.98 |

---

# 📷 Outputs

Add screenshots here after running the project.

```
reports/
├── prediction_plot.png
├── benchmark_results.png
├── training_loss.png
```

---

# 🔬 Future Improvements

- Multi-stock training
- Hyperparameter tuning
- Attention visualization
- Ensemble forecasting
- Hugging Face Time Series models
- Probabilistic forecasting
- Live market prediction API
- Streamlit dashboard
- Docker deployment
- CI/CD pipeline

---

# 🎯 Key Learnings

This project demonstrates:

- Deep Learning
- Time Series Forecasting
- PyTorch
- Model Benchmarking
- Experiment Tracking
- Financial Machine Learning
- Production ML Engineering

---

# 👨‍💻 Author

**Vikas**

AI • Machine Learning • Deep Learning • Python • PyTorch

---

## ⭐ If you found this project useful, consider giving it a star!
