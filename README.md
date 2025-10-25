# ML Predictive Maintenance Project ⚙️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![GitHub](https://img.shields.io/badge/Repo-GitHub-lightgrey.svg)

This repository contains the full workflow for an industrial Predictive Maintenance (PdM) solution. The primary objective is to leverage machine learning and deep learning to analyze complex sensor data, detect degradation patterns, and forecast potential equipment failures before they occur.

## 🧠 Project Title
Development of a Predictive Maintenance Model Based on Machine Learning

### Project Description
The goal of this project is to develop a predictive maintenance model that can anticipate failures of industrial equipment before they occur. The work begins by building a representative dataset from scratch, either synthesized or collected, covering key operational parameters of machines (temperature, vibration, pressure, electric current, etc.), failure logs, and performance indicators.

An exploratory data analysis (EDA) phase is then conducted to discover trends, correlations, and anomalies. Based on this analysis, various machine learning models (Random Forest, XGBoost, LightGBM, SVM, etc.) are tested to predict:
- The most probable type of fault (classification)
- And/or the time of failure occurrence (time series regression)

The model performances are evaluated using appropriate metrics (accuracy, F1-score, RMSE, etc.), allowing the selection of the most reliable solution for potential integration into an intelligent monitoring system.

**Technologies and Tools:**
- Language: Python 3.x
- Libraries: pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn
- Environment: Jupyter Notebook, VS Code
- Application Domain: Industry 4.0 – Predictive Maintenance

## 🎯 Main Objectives

This project is designed to optimize industrial operations by moving from reactive or preventive maintenance to a proactive, predictive strategy.

* Optimize Maintenance: Anticipate failures to enable proactive interventions and scheduled repairs.
* Reduce Downtime: Minimize costly operational stoppages by forecasting faults early.
* Understand Sensor Behavior: Link variations in sensor readings (temperature, vibration, etc.) to specific equipment degradation patterns.
* Develop Scalable Models: Build and validate robust ML/DL models deployable in industrial settings (Edge/Cloud).
* Ensure Reproducibility: Maintain a clear, version-controlled environment (Jupyter + GitHub) for experimentation and collaboration.

## 🛠️ Core Capabilities

This project framework covers the complete end-to-end ML pipeline:

1. Dataset Creation from Scratch:
   * Build a comprehensive dataset reflecting industrial scenarios and sensor readings.
   * Ensure coverage of machine parameters, failure history, and KPIs.
2. Industrial Data Analysis & Processing:
   * Handle multivariate time-series sensor datasets (temperature, vibration, current, torque, etc.).
   * Detect patterns, correlations, and anomalies in machine behavior.
3. Advanced Data Exploration & Visualization:
   * Generate clear analytical visuals (heatmaps, trend lines, anomaly plots) using Matplotlib and Seaborn.
4. Machine Failure Forecasting (ML/DL):
   * Develop and train models for:
      * Classification: "Will this machine fail in the next N hours?"
      * Regression: "What is the Remaining Useful Life (RUL) of this component?"
5. AI Model Development & Optimization:
   * Implement flexible models ranging from classical ML (Random Forest, XGBoost, LightGBM, SVM) to neural networks (LSTM, CNNs) using Scikit-learn, TensorFlow, and PyTorch.
6. Full ML Pipeline Integration:
   * Perform preprocessing, normalization, feature engineering (e.g., rolling averages, frequency analysis), model training, cross-validation, and performance evaluation.

## 💻 Technology Stack

| Domain              | Tools & Frameworks                          | Description                                           |
| :------------------| :-------------------------------------------| :---------------------------------------------------- |
| Language           | Python 3.x                                  | Core language for data science and ML.                |
| Data Handling      | pandas, NumPy                               | Data cleaning, transformation, and feature engineering |
| Visualization      | Matplotlib, Seaborn                         | Visual exploration, trend analysis, results plotting   |
| Machine Learning   | Scikit-learn, XGBoost, LightGBM, SVM        | Classification, regression, and performance evaluation |
| Deep Learning      | TensorFlow, Keras, PyTorch                  | Neural network modeling for advanced time-series       |
| Environment        | Jupyter Notebook, VS Code                   | Experimentation, prototyping, dataset integration      |
| Version Control    | Git, GitHub                                 | Workflow management, tracking, collaboration           |

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mehdi007bond/Predictive_maintenance_Project.git
cd Predictive_maintenance_Project
```

### 2. Set Up a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: You need to create a requirements.txt using `pip freeze > requirements.txt` after installing dependencies)*

### 4. Run the Project

```bash
jupyter notebook
```

Open the notebooks/ directory and run the notebooks in order (e.g., 01_Data_Generation.ipynb, 02_Data_Exploration.ipynb, etc.)

## 📂 Project Structure (Example)

```
.
├── notebooks/
│   ├── 01_Data_Generation.ipynb
│   ├── 02_Data_Exploration.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Training_Classification.ipynb
│   ├── 05_Model_Training_Regression_RUL.ipynb
│   └── 06_Model_Evaluation.ipynb
│
├── data/
│   ├── raw/
│   │   └── synthetic_maintenance_data.csv
│   └── processed/
│       └── featured_data.parquet
│
├── src/
│   ├── preprocessing.py     # Functions for data cleaning
│   └── feature_engineering.py # Functions for creating new features
│
├── models/
│   ├── classification_model.pkl   # Saved classifier
│   └── regression_rul_model.h5    # Saved RUL (NN) model
│
├── .gitignore
├── README.md                # This file
└── requirements.txt         # Project dependencies
```