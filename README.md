# ML Predictive Maintenance Project ⚙️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![GitHub](https://img.shields.io/badge/Repo-GitHub-lightgrey.svg)

This repository contains the full workflow for an industrial Predictive Maintenance (PdM) solution. The primary objective is to leverage machine learning and deep learning to analyze complex sensor data, detect degradation patterns, and forecast potential equipment failures *before* they occur.

## 🎯 Main Objectives

This project is designed to optimize industrial operations by moving from reactive/preventive maintenance to a proactive, *predictive* strategy.

* **Optimize Maintenance:** Anticipate failures to enable proactive interventions and scheduled repairs.
* **Reduce Downtime:** Minimize costly operational stoppages by forecasting faults early.
* **Understand Sensor Behavior:** Link variations in sensor readings (temperature, vibration, etc.) to specific equipment degradation patterns.
* **Develop Scalable Models:** Build and validate robust ML/DL models that can be deployed in real industrial environments (Edge/Cloud).
* **Ensure Reproducibility:** Maintain a clear, version-controlled environment (Jupyter + GitHub) for experimentation and collaboration.

## 🛠️ Core Capabilities

This project framework covers the complete end-to-end ML pipeline:

1.  **Industrial Data Analysis & Processing:**
    * Handles complex, multi-variate time-series sensor datasets (e.g., temperature, vibration, current, torque).
    * Detects patterns, correlations, and anomalies in machine behavior.

2.  **Advanced Data Exploration & Visualization:**
    * Generates clear analytical visuals (heatmaps, trend lines, anomaly plots) using Matplotlib and Seaborn to build domain understanding.

3.  **Machine Failure Forecasting (ML/DL):**
    * Builds and trains models for both:
        * **Classification:** "Will this machine fail in the next N hours?"
        * **Regression:** "What is the Remaining Useful Life (RUL) of this component?"

4.  **AI Model Development & Optimization:**
    * Implements flexible models ranging from classical ML (Random Forest, XGBoost) to Neural Networks (LSTM, CNNs) using Scikit-learn, TensorFlow, and PyTorch.

5.  **Full ML Pipeline Integration:**
    * Covers preprocessing, normalization, advanced feature engineering (e.g., rolling averages, frequency analysis), model training, cross-validation, and performance evaluation.

## 💻 Technology Stack

| Domain | Tools & Frameworks | Description |
| :--- | :--- | :--- |
| **Language** | `Python 3.x` | Core language for data science and ML. |
| **Data Handling** | `pandas`, `NumPy` | Data cleaning, transformation, and feature engineering. |
| **Visualization** | `Matplotlib`, `Seaborn` | Visual exploration, trend analysis, and results plotting. |
| **Machine Learning**| `Scikit-learn`, `XGBoost` | Classification, regression, and performance evaluation. |
| **Deep Learning** | `TensorFlow`, `Keras`, `PyTorch` | Neural network modeling for advanced time-series prediction. |
| **Environment** | `Jupyter Notebook`, `Kaggle` | Experimentation, prototyping, and dataset integration. |
| **Version Control**| `Git`, `GitHub` | Workflow management, tracking, and collaboration. |

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Mehdi007bond/Predictive_maintenance_Project.git
cd Predictive_maintenance_Project
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required packages listed in the requirements.txt file.

```bash
pip install -r requirements.txt
```

*(Note: You will need to create a requirements.txt file. You can generate one from your environment using `pip freeze > requirements.txt`)*

### 4. Run the Project

The core analysis and model development are structured within Jupyter Notebooks.
Start Jupyter:

```bash
jupyter notebook
```

Follow the Notebooks: Open the notebooks/ directory and run the notebooks in sequential order (e.g., `01_Data_Exploration.ipynb`, `02_Feature_Engineering.ipynb`, etc.)

## 📂 Project Structure (Example)

A recommended structure for this project:

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
