# Multi-Layer Intrusion Detection System

A machine learning project to detect **SYN scanning** (Network Layer) and **SQL Injection** (Application Layer) attacks.

---

## 📋 Table of Contents

1. [System Overview](#-system-overview)
2. [Prerequisites](#-prerequisites)
3. [Installation](#-installation)
4. [How to Run](#-how-to-run)
5. [Project Structure](#-project-structure)
6. [Detection Methods](#-detection-methods)
7. [Configuration](#-configuration)
8. [Evaluation Metrics](#-evaluation-metrics)
9. [Troubleshooting](#-troubleshooting)

---

## 🎯 System Overview

This is a **multi-layer IDS** that combines two independent ML models:

| Layer | Attack | Detection Method |
|-------|--------|------------------|
| **Network (L3/L4)** | SYN Scan | TCP flag behavior analysis |
| **Application (L7)** | SQL Injection | HTTP payload pattern analysis |

---

## ⚙️ Prerequisites

Before running the project, ensure you have:

- **Python 3.8+** installed on your system
- **pip** (Python package installer)
- Raw network packet data in CSV format (place in `data/raw/` folder)

---

## 📦 Installation

### Step 1: Clone/Navigate to the Project

```bash
cd "Multi-Layer Intrusion Detection System"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages:
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting (optional)
- `matplotlib` & `seaborn` - Visualization
- `pyyaml` - Configuration handling
- `streamlit` - Web interface (for `app.py`)
- `plotly` - Interactive charts

---

## 🚀 How to Run

### Option 1: Train Models from Scratch

#### 1A. Train SYN Scan Detection Model

```bash
python main.py
```

This will:
1. Load raw packet data from `data/raw/project_features_raw9.0.csv`
2. Preprocess and aggregate packets into time windows
3. Extract features and create labels
4. Train a Random Forest classifier
5. Evaluate model performance
6. Save the model to `models/syn_scan_detector.joblib`

#### 1B. Train SQL Injection Detection Model

```bash
python main_sqli.py
```

This will:
1. Extract HTTP requests from raw data
2. Decode URI payloads
3. Extract lexical features (SQL keywords, comments, special chars)
4. Train a Random Forest classifier
5. Save the model to `models/sqli_detector.joblib`

---

### Option 2: Run Predictions on New Data

Once models are trained, use `predict.py` to analyze new network captures:

```bash
python predict.py <path_to_raw_csv>
```

**Example:**
```bash
python predict.py data/raw/new_capture.csv
```

---

### Option 3: Launch Web Interface (Streamlit App)

For an interactive web dashboard:

```bash
python -m streamlit run app.py
```

This will:
1. Open a browser at `http://localhost:8501`
2. Allow you to upload CSV files for analysis
3. Display detection results with visualizations
4. Show attack statistics and alerts

---

## 📁 Project Structure

```
Multi-Layer Intrusion Detection System/
├── configs/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/                     # Raw packet data (CSV)
│   │   └── project_features_raw9.0.csv
│   └── processed/               # Processed features
│       ├── aggregated_features_labeled.csv  # SYN scan data
│       └── http_features_labeled.csv        # SQL injection data
├── models/
│   ├── syn_scan_detector.joblib  # Trained SYN scan model
│   └── sqli_detector.joblib      # Trained SQL injection model
├── reports/
│   ├── A-Report.pdf              # Model evaluation report
│   └── figures/                  # Evaluation plots
├── src/
│   ├── data/
│   │   ├── preprocess.py         # SYN scan preprocessing
│   │   └── preprocess_sqli.py    # SQL injection preprocessing
│   ├── features/
│   │   ├── build_features.py     # SYN scan feature engineering
│   │   └── build_features_sqli.py # SQL injection features
│   ├── models/
│   │   ├── train.py              # SYN scan model training
│   │   ├── train_sqli.py         # SQL injection model training
│   │   └── predict.py            # Prediction utilities
│   ├── evaluation/
│   │   ├── evaluate.py           # SYN scan evaluation
│   │   └── evaluate_sqli.py      # SQL injection evaluation
│   └── utils/
│       └── io.py                 # I/O utilities #Not used 
├── main.py                       # SYN scan detection pipeline
├── main_sqli.py                  # SQL injection detection pipeline
├── predict.py                    # Prediction script for new data
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔍 Detection Methods

### 🔹 1. SYN Scan Detection (Network Layer)

**Key Concept:** SYN scanning cannot be detected from a single packet!

Instead, temporal patterns across multiple packets are analyzed using time-window-based aggregation.

#### What makes SYN scan detectable?

| Feature | Normal Traffic | SYN Scan |
|---------|---------------|----------|
| SYN packets | Few | Many |
| ACK packets | Many | Very few |
| Ports | Same | Many different |
| Time delta | Large | Very small |
| Connection completion | Yes | No |

---

### 🔹 2. SQL Injection Detection (Application Layer)

**Key Concept:** SQL injection is detected by analyzing HTTP request payloads for malicious patterns.

#### What makes SQL injection detectable?

| Feature | Normal Request | SQL Injection |
|---------|---------------|---------------|
| SQL keywords | None | SELECT, UNION, DROP |
| Comments | None | --, #, /* */ |
| Boolean tricks | None | OR 1=1, OR 'a'='a' |
| Special chars | Few | Many (', ", ;, =) |
| Payload length | Short | Often long |

---

## 📊 SYN Scan ML Approach

### Time-Window Based Aggregation

Packets are grouped by `(src_ip, dst_ip, time_window)`:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `syn_count` | Number of SYN packets | High in SYN scan |
| `ack_count` | Number of ACK packets | Low in SYN scan |
| `syn_ack_ratio` | SYN / ACK ratio | **Key indicator!** |
| `unique_dst_ports` | Unique destination ports | High = port scanning |

### Labeling Logic

```python
(syn_count > 5 AND unique_ports > 3) OR (syn_ack_ratio > 3.0)
```

---

## 📊 SQL Injection ML Approach

### Feature Extraction from HTTP Payloads

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `has_sql_keywords` | Contains SELECT, UNION, etc. | **Key indicator!** |
| `has_comment` | Contains --, #, /* | Attack obfuscation |
| `has_or_true` | Contains OR 1=1 | Boolean injection |
| `special_char_count` | Count of ', ", ; | Escape attempts |
| `keyword_count` | Number of SQL keywords | Attack complexity |

### Labeling Logic

```python
(has_sql_keywords AND special_chars > 2) OR has_or_true OR (has_comment AND has_keywords)
```

---

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
aggregation:
  time_window_seconds: 1  # Window size for packet aggregation
  min_packets_per_window: 2

labeling:
  syn_count_threshold: 5       # Min SYN packets for attack
  unique_ports_threshold: 3    # Min ports for port scan
  syn_ack_ratio_threshold: 3.0 # SYN/ACK ratio threshold

model:
  type: "random_forest"  # Options: "random_forest", "xgboost"
```

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted attacks, how many are real?
- **Recall**: Of actual attacks, how many did we detect? ⭐
- **F1 Score**: Balance of precision and recall

For security applications, **high recall** is critical (don't miss attacks!).

---

## ❓ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Raw data file not found` | Place your CSV in `data/raw/` folder |
| `Model not found` | Run `main.py` and `main_sqli.py` first to train models |
| `No HTTP requests found` | Ensure your data contains HTTP traffic columns |
| Streamlit not starting | Install with `pip install streamlit plotly` |

### Data Requirements

Your raw CSV should contain these columns:
- `frame.time_epoch` - Timestamp
- `ip.src`, `ip.dst` - IP addresses
- `tcp.srcport`, `tcp.dstport` - Ports
- `tcp.flags.syn`, `tcp.flags.ack` - TCP flags
- `http.request.uri` - HTTP URIs (for SQLi detection)

---

## 📝 Report Explanation

Use this in your report:

> "A multi-layer intrusion detection system was implemented using two independent machine learning models. The first model detects SYN scanning attacks by analyzing aggregated TCP traffic features within time windows. The second model identifies SQL injection attacks by examining lexical and statistical features of HTTP request payloads. A final alert engine correlates the outputs of both models to detect network and application-layer attacks effectively."

---

## 👥 Authors

Network Security Project Team



