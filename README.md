# 🔧 ML Predictive Maintenance Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-69%2C120%20samples-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehdi007bond/Predictive_maintenance_Project/blob/main/Predictive_maintenance_Project_.ipynb)

## 📋 Overview

This repository implements a comprehensive **Predictive Maintenance (PdM)** solution using machine learning and deep learning techniques. The project covers the complete pipeline from synthetic data generation to LSTM-based failure prediction for industrial equipment monitoring.

### 🎯 Key Features
- **Synthetic Dataset Generation**: 69,120 time-series samples across 4 production lines and 12 machines
- **Multi-Machine Types**: Fraiseuse, Convoyeur, Machine_de_finition with distinct degradation patterns
- **LSTM Neural Networks**: Deep learning approach for sequence-based failure prediction
- **Comprehensive EDA**: Statistical analysis and visualization of sensor patterns
- **Real-time Prediction**: State-based classification (failure=1 within prediction horizon)

## 🏭 Problem Statement

Industrial equipment failures lead to costly downtime and safety hazards. This project develops a predictive maintenance system that:

1. **Monitors** multi-sensor data (temperature, vibration, current, torque)
2. **Detects** degradation patterns using LSTM neural networks
3. **Predicts** failure states 58 hours in advance (4% positive rate)
4. **Prevents** unplanned downtime through early intervention

## 🧠 Deep Learning Architecture (LSTM)

### Problem Framing
- **Classification Task**: Detect failing-state windows (failure=1) within prediction horizon
- **Sequence Modeling**: Use rolling windows of sensor readings to capture temporal dependencies
- **Multi-sensor Input**: [temperature, vibration, current, torque, total_working_hours]

### Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(window_size, n_features)),
    LSTM(32),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

### Training Configuration
- **Window Size**: 64 timesteps (16 hours at 4 samples/hour)
- **Features**: 5 standardized sensor readings
- **Class Weighting**: Address 4% positive class imbalance
- **Optimization**: Adam optimizer with ReduceLROnPlateau
- **Early Stopping**: Patience=5, restore best weights
- **Validation**: Time-based split to prevent data leakage

## 📊 Dataset Specifications

| Parameter | Value |
|-----------|-------|
| **Total Samples** | 69,120 |
| **Production Lines** | 4 (Line_1 to Line_4) |
| **Machines per Line** | 3 (Fraiseuse, Convoyeur, Machine_de_finition) |
| **Sampling Rate** | 4 samples/hour |
| **Duration** | 60 days per machine |
| **Failure Rate** | ~4% (2,784 positive samples) |
| **Features** | timestamp, production_line, machine_id, machine_type, temperature, vibration, current, torque, total_working_hours, failure |

### Machine Profiles
Each machine type has distinct operational characteristics:

- **Fraiseuse**: High temperature/vibration, intensive operations
- **Convoyeur**: Moderate temperature, current-sensitive
- **Machine_de_finition**: Low vibration, precision operations

## 🔄 Reproduce LSTM Training

### 1. Environment Setup
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn
```

### 2. Data Preparation
```python
# Load and prepare data
df = pd.read_csv('production_line_STATE_BASED_4_PERCENT_data.csv')
features = ['temperature', 'vibration', 'current', 'torque', 'total_working_hours']

# Sort by machine_id and timestamp
df_sorted = df.sort_values(['machine_id', 'timestamp'])
```

### 3. Sequence Windowing
```python
def create_sequences(data, window_size=64, stride=1):
    X, y = [], []
    for i in range(0, len(data) - window_size + 1, stride):
        X.append(data[i:(i + window_size), :])
        # Label by max failure in window or failure at window end
        y.append(max(labels[i:(i + window_size)]))
    return np.array(X), np.array(y)
```

### 4. Train/Validation Split
```python
# Time-based split to prevent leakage
split_date = '2024-02-15'
train_data = df[df['timestamp'] < split_date]
val_data = df[df['timestamp'] >= split_date]
```

### 5. Model Training
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights for imbalanced data
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(y_train), 
                                   y=y_train)

# Train with early stopping
history = model.fit(X_train, y_train,
                   validation_data=(X_val, y_val),
                   epochs=50,
                   batch_size=256,
                   class_weight=dict(enumerate(class_weights)),
                   callbacks=[early_stopping, lr_reduction])
```

### 6. Evaluation Metrics
- **AUROC**: Area under ROC curve for binary classification
- **AUPRC**: Area under Precision-Recall curve (important for imbalanced data)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision@K**: Precision at top K predictions (alert budget)
- **Per-segment Analysis**: Performance breakdown by production line and machine type

## 💻 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.x | Core development |
| **Data Processing** | pandas, NumPy | Data manipulation and analysis |
| **Deep Learning** | TensorFlow, Keras | LSTM neural networks |
| **ML Algorithms** | Scikit-learn, XGBoost | Traditional ML models |
| **Visualization** | Matplotlib, Seaborn | Data exploration and results |
| **Environment** | Jupyter Notebook, Google Colab | Interactive development |
| **Version Control** | Git, GitHub | Code management |

## 📈 Getting Started

### Clone Repository
```bash
git clone https://github.com/Mehdi007bond/Predictive_maintenance_Project.git
cd Predictive_maintenance_Project
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Notebooks
1. **Data Generation**: Execute `Predictive_maintenance_Project_.ipynb` to generate synthetic dataset
2. **Exploratory Analysis**: Analyze sensor patterns and failure distributions
3. **LSTM Training**: Implement sequence modeling for failure prediction
4. **Model Evaluation**: Assess performance and generate insights

## 📂 Project Structure

```
Predictive_maintenance_Project/
├── 📄 Predictive_maintenance_Project_.ipynb  # Main notebook
├── 📊 production_line_STATE_BASED_4_PERCENT_data.csv  # Generated dataset
├── 🔧 src/
│   ├── data_generation.py      # Synthetic data creation
│   ├── sequence_modeling.py    # LSTM windowing and training
│   └── evaluation.py           # Model assessment utilities
├── 📈 models/
│   ├── lstm_failure_predictor.h5    # Trained LSTM model
│   └── feature_scaler.pkl           # StandardScaler for preprocessing
├── 📋 requirements.txt         # Project dependencies
└── 📖 README.md               # This file
```

## 🔬 Methodology

### 1. Data Generation
- **Synthetic Dataset**: Realistic sensor data with controllable failure patterns
- **Machine Profiles**: Distinct parameter ranges for different equipment types
- **Temporal Degradation**: Progressive sensor deterioration before failures
- **Balanced Scenarios**: 4% failure rate matching real-world distributions

### 2. Exploratory Data Analysis (EDA)
- **Sensor Correlation**: Identify relationships between parameters
- **Failure Patterns**: Analyze degradation signatures per machine type
- **Temporal Trends**: Visualize sensor evolution over operational hours
- **Statistical Distribution**: Characterize normal vs. failure states

### 3. LSTM Implementation
- **Sequence Windows**: 64-timestep windows for temporal pattern recognition
- **Feature Scaling**: StandardScaler normalization for stable training
- **Architecture**: Bidirectional LSTM layers with dropout regularization
- **Class Balancing**: Weighted loss function for imbalanced data
- **Validation**: Time-based splits to prevent data leakage

### 4. Model Evaluation
- **Performance Metrics**: AUROC, AUPRC, F1-score, Precision@K
- **Segment Analysis**: Per production line and machine type breakdown
- **Calibration**: Reliability of predicted probabilities
- **Alert Optimization**: Threshold selection for operational deployment

## 📊 Expected Results

- **High Recall**: Minimize missed failures (false negatives)
- **Controlled Precision**: Balance alert fatigue with safety requirements
- **Early Detection**: 58-hour prediction horizon for maintenance planning
- **Interpretability**: Understanding of sensor-failure relationships
- **Scalability**: Framework applicable to different industrial contexts

## 🔮 Future Enhancements

- **Anomaly Detection**: Unsupervised learning for unknown failure modes
- **Multi-task Learning**: Joint prediction of failure type and RUL
- **Federated Learning**: Privacy-preserving training across multiple sites
- **Real-time Deployment**: Edge computing integration with IoT sensors
- **Explanation AI**: SHAP/LIME analysis for model interpretability

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out through:
- GitHub Issues: [Project Issues](https://github.com/Mehdi007bond/Predictive_maintenance_Project/issues)
- LinkedIn: [Your LinkedIn Profile]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⚙️ **Built with passion for Industrial AI and Predictive Analytics** ⚙️

*Transforming reactive maintenance into intelligent, proactive solutions through machine learning and deep learning innovations.*