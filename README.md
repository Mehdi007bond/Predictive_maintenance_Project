# ML Predictive Maintenance Project ⚙️

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
![GitHub](https://img.shields.io/badge/Repo-GitHub-lightgrey.svg)

This repository contains the full workflow for an industrial Predictive Maintenance (PdM) solution. The primary objective is to leverage machine learning and deep learning to analyze complex sensor data, detect degradation patterns, and forecast potential equipment failures *before* they occur.

## 🧠 Project Title (Titre du Projet)
**Développement d’un modèle de maintenance prédictive basé sur le Machine Learning**

### Description du projet :
Ce projet a pour objectif de développer un modèle de maintenance prédictive capable d’anticiper les défaillances des équipements industriels avant qu’elles ne se produisent. Il s’inscrit dans une démarche d’optimisation de la performance opérationnelle et de réduction des coûts de maintenance au sein d’un environnement industriel digitalisé.

Le travail débute par **la constitution d’un dataset représentatif à partir de zéro**, construit à partir de données simulées ou collectées. Ce dataset comprend les paramètres essentiels du fonctionnement des machines (température, vibration, pression, courant électrique, etc.), les historiques de pannes et les indicateurs de performance.

Une phase d’analyse exploratoire des données (EDA) est ensuite menée afin d’identifier les tendances, corrélations et anomalies. Sur cette base, différents modèles d’apprentissage automatique (Random Forest, XGBoost, LightGBM, SVM…) seront testés pour prédire :
- Le type de panne probable (classification)
- Et/ou le moment de survenue d’une défaillance (régression temporelle)

Enfin, les performances des modèles seront évaluées à l’aide de métriques adaptées (précision, F1-score, RMSE…), afin de sélectionner la solution la plus fiable pour une intégration potentielle dans un système de supervision intelligent.

**Technologies et outils :**
- Langage : Python
- Bibliothèques : Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn
- Environnement : Jupyter Notebook / Visual Studio Code
- Domaine d’application : Industrie 4.0 – Maintenance prédictive

## 🎯 Main Objectives

This project is designed to optimize industrial operations by moving from reactive/preventive maintenance to a proactive, *predictive* strategy.

* **Optimize Maintenance:** Anticipate failures to enable proactive interventions and scheduled repairs.
* **Reduce Downtime:** Minimize costly operational stoppages by forecasting faults early.
* **Understand Sensor Behavior:** Link variations in sensor readings (temperature, vibration, etc.) to specific equipment degradation patterns.
* **Develop Scalable Models:** Build and validate robust ML/DL models that can be deployed in real industrial environments (Edge/Cloud).
* **Ensure Reproducibility:** Maintain a clear, version-controlled environment (Jupyter + GitHub) for experimentation and collaboration.

## 🛠️ Core Capabilities

This project framework covers the complete end-to-end ML pipeline:

1.  **Dataset Creation from Scratch:**
    * Builds a realistic synthetic or collected dataset representing industrial behavior and sensor readings.
    * Establishes data collection integrity with machine parameters, failure history, and key KPIs.

2.  **Industrial Data Analysis & Processing:**
    * Handles complex, multi-variate time-series sensor datasets (e.g., temperature, vibration, current, torque).
    * Detects patterns, correlations, and anomalies in machine behavior.

3.  **Advanced Data Exploration & Visualization:**
    * Generates clear analytical visuals (heatmaps, trend lines, anomaly plots) using Matplotlib and Seaborn to build domain understanding.

4.  **Machine Failure Forecasting (ML/DL):**
    * Builds and trains models for both:
        * **Classification:** "Will this machine fail in the next N hours?"
        * **Regression:** "What is the Remaining Useful Life (RUL) of this component?"

5.  **AI Model Development & Optimization:**
    * Implements flexible models ranging from classical ML (Random Forest, XGBoost) to Neural Networks (LSTM, CNNs) using Scikit-learn, TensorFlow, and PyTorch.

6.  **Full ML Pipeline Integration:**
    * Covers preprocessing, normalization, advanced feature engineering (e.g., rolling averages, frequency analysis), model training, cross-validation, and performance evaluation.

## 💻 Technology Stack

| Domain | Tools & Frameworks | Description |
| :--- | :--- | :--- |
| **Language** | `Python 3.x` | Core language for data science and ML. |
| **Data Handling** | `pandas`, `NumPy` | Data cleaning, transformation, and feature engineering. |
| **Visualization** | `Matplotlib`, `Seaborn` | Visual exploration, trend analysis, and results plotting. |
| **Machine Learning**| `Scikit-learn`, `XGBoost`, `LightGBM` | Classification, regression, and performance evaluation. |
| **Deep Learning** | `TensorFlow`, `Keras`, `PyTorch` | Neural network modeling for advanced time-series prediction. |
| **Environment** | `Jupyter Notebook`, `Kaggle`, `VS Code` | Experimentation, prototyping, and dataset integration. |
| **Version Control**| `Git`, `GitHub` | Workflow management, tracking, and collaboration. |

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

*(Note: You will need to create a requirements.txt file. You can generate one from your environment using `pip freeze > requirements.txt`)*

### 4. Run the Project

```bash
jupyter notebook
```

Follow the Notebooks: Open the notebooks/ directory and run the notebooks in sequential order (e.g., `01_Data_Generation.ipynb`, `02_Data_Exploration.ipynb`, etc.)

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