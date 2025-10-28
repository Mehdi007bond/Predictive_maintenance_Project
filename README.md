# 🔧 Predictive Maintenance Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-blueviolet.svg)
![Dataset](https://img.shields.io/badge/Dataset-69%2C120%20samples-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mehdi007bond/Predictive_maintenance_Project/blob/main/Predictive_maintenance_Project_.ipynb)

## Overview
This project simulates industrial sensor data across multiple production lines and machine types to study failure patterns and support predictive maintenance. It programmatically generates a balanced time-series dataset with a realistic 4% failing-state prevalence, then performs exploratory analysis and visualization to understand sensor behavior during degradation.

## 🎯 Purpose
- Anticipate failures with sufficient lead time (~58 hours) to plan maintenance
- Reduce unplanned downtime and optimize maintenance schedules
- Provide a reproducible baseline (data generation + future LSTM modeling) ready for industrial adaptation
- Study degradation patterns across different machine types and production lines

## 🛠 Technologies Used
- **Python 3.x** - Core programming language
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Basic plotting and visualization
- **seaborn** - Statistical data visualization
- **datetime** - Time series handling (built-in)
- **Jupyter/Colab** - Interactive development environment
- **TensorFlow/Keras** - (Planned for LSTM implementation)

## 📊 What This Project Does
- **Synthetic data generation** for 4 production lines and 12 machines over 60 days at 4 samples/hour
- **Failure-state simulation** with realistic, progressive sensor degradation beginning 58 hours before failure
- **Dataset export** to CSV format (`production_line_STATE_BASED_4_PERCENT_data.csv`)
- **Exploratory data analysis (EDA)** with histograms, distributions, and statistical analysis
- **Per-line and per-machine-type** slicing and visualization
- **Future: LSTM-based sequence modeling** for failure prediction with temporal windows

## 📋 Dataset Design

### Entities
- **4 production lines**: Line_1, Line_2, Line_3, Line_4
- **3 machine types per line**: Fraiseuse, Convoyeur, Machine_de_finition
- **12 total machines** (4×3)
- **Total records**: 60 days × 24 hours × 4 samples/hour × 12 machines = **69,120 rows**

### Columns
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Sample timestamp |
| `production_line` | string | Line_1..Line_4 |
| `machine_id` | int | 1..12 |
| `machine_type` | string | Fraiseuse \| Convoyeur \| Machine_de_finition |
| `temperature` | float | Temperature (°C) |
| `vibration` | float | Vibration measurement |
| `current` | float | Current (A) |
| `torque` | float | Torque (Nm) |
| `total_working_hours` | float | Cumulative operating hours |
| `failure` | int | 0 (healthy), 1 (failing-state window) |

### Failure Modeling
- **Target prevalence**: ~4% failing-state (2,784 samples out of 69,120)
- **Failing state window**: Starts 58 hours before randomly selected failure point
- **Progressive degradation** during failing window:
  - **Fraiseuse**: temperature ↑, vibration ↑, current ↑, torque ↑
  - **Convoyeur**: temperature ↑, vibration ↑, current ↑, torque ↑
  - **Machine_de_finition**: temperature ↑, vibration ↑, current ↓, torque ↓

## 📁 Notebook Structure (Key Steps)

### 1. Configuration and Parameters
```python
N_LINES = 4
DAYS_PER_MACHINE = 60
SAMPLES_PER_HOUR = 4
FAILURE_STATE_HOURS = 58
```

### 2. Machine Profiles
- Base values and noise parameters per machine type
- Type-specific degradation rates applied during failing-state interval
- Temperature, vibration, current, torque parameter ranges

### 3. Data Generation Loop
- Builds per-machine time series with slight timestamp offset (seconds)
- Assigns failure window and applies progressive degradation
- Concatenates all machine DataFrames and sorts globally by timestamp
- Exports to `production_line_STATE_BASED_4_PERCENT_data.csv`

### 4. Exploratory Data Analysis (EDA)
- `df.info()`, `df.describe()`, `df.shape` - Basic dataset overview
- Histograms for all numeric columns (`df.hist()`)
- Simple per-line and per-type data slicing
- Distribution analysis and failure state visualization

## 📈 Quick Results
- **Total rows**: 69,120
- **Failure distribution**: 66,336 healthy (96%), 2,784 failing-state (4%)
- **Columns**: 10 core features
- **Time span**: 60 days per machine
- **Sampling rate**: 4 samples/hour
- **CSV file size**: ~2MB generated dataset

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Mehdi007bond/Predictive_maintenance_Project.git
cd Predictive_maintenance_Project
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### 3. Launch Notebook
```bash
jupyter notebook Predictive_maintenance_Project_.ipynb
```
*Or open directly in Google Colab using the badge at the top of the notebook*

## 🧠 Method Overview (Current + Planned)
1. **Data Generation**: Realistic sensor signals with machine-specific degradation profiles
2. **EDA**: Understanding distributions, correlations, and failure patterns  
3. **Sequence Modeling (Planned)**: LSTM with temporal windows to classify failing states
4. **Evaluation (Planned)**: AUROC/AUPRC/F1 metrics and per-segment analysis (line/type)

## ⚠️ Notes, Caveats, and Tips
- **Failure label** represents a state window before failure (classification framing), not point failure or RUL; models should be designed accordingly
- **Degradation patterns** are deliberately simple and type-dependent to make patterns detectable and explainable
- **Dataset balance** (4% failing-state) is optimized for study rather than real-world skew; adjust `FAILURE_STATE_HOURS` or failure sampling logic for different targets
- **Timestamp alignment**: Slight random offsets prevent perfect synchronization between machines
- **CSV output**: Generated dataset is automatically saved and can be reused without re-running generation

## 🛠 Future Work (Roadmap)
- [ ] **Feature Engineering**: Add train/validation splits and rolling window features
- [ ] **LSTM Implementation**: Build sequence models with temporal windows for failure classification
- [ ] **Baseline Models**: Implement logistic regression and random forest for comparison
- [ ] **Time-series Features**: Sliding windows, statistical aggregations, trend analysis
- [ ] **Model Evaluation**: AUROC, AUPRC, F1-score with calibration analysis
- [ ] **RUL Extension**: Extend to Remaining Useful Life regression framing
- [ ] **Real-time Monitoring**: Add dashboards and streaming prediction capabilities
- [ ] **Industrial Deployment**: Edge computing and IoT integration considerations

## 📁 Repository Layout (Current)
```
Predictive_maintenance_Project/
├── Predictive_maintenance_Project_.ipynb    # Main notebook with data generation and EDA
├── production_line_STATE_BASED_4_PERCENT_data.csv    # Generated dataset (output)
├── README.md                                # This documentation
└── requirements.txt                         # Python dependencies (planned)
```

## 👨‍💻 Author
**Mehdi Boumazzourh** ([@Mehdi007bond](https://github.com/Mehdi007bond))

## 📜 License
MIT License - see LICENSE file for details

---
*This project provides a foundation for predictive maintenance research and can be adapted for real industrial applications. The current implementation focuses on data generation and EDA, with LSTM modeling planned as the next major milestone.*