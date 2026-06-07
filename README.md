# Data Classification Using AI

## 📌 Project Overview
This project is part of an Industrial Training Kit (Batch 2026). The goal of this project is to build a complete Supervised Machine Learning pipeline from scratch using the Input-Process-Output (IPO) framework. 

It uses the **K-Nearest Neighbors (KNN)** algorithm to classify the species of Iris flowers based on physical measurements.

## ⚙️ Tech Stack
* **Language:** Python 3.11
* **Libraries:** `scikit-learn`, `numpy`
* **Algorithm:** K-Nearest Neighbors (KNN)

## 🏗️ Architecture (IPO Framework)
1. **Input:** Loaded the Iris Benchmark dataset and performed an 80/20 train-test split, shuffling the data to prevent order bias.
2. **Process:** Applied `StandardScaler` to normalize features (preventing distance bias) and trained the KNN model (`K=5`). Ensured no data leakage by strictly using `.transform()` on the testing data.
3. **Output:** Evaluated the model against unseen data using a Confusion Matrix and Macro F1 Score.

## 🚀 How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/louayakermi2816-gif/Data-Classification-Using-AI.git
   cd Data-Classification-Using-AI
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate       # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python main.py
   ```