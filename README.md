# Network Intrusion Detection System

Production-ready multi-class intrusion detection achieving 98% macro-F1 on CICIDS2017 dataset (2.2M flows, 10 attack types).

## Problem

Security operations centers need systems that:
- Detect diverse attacks (DDoS, web attacks, botnet, protocol exploits)
- Handle extreme class imbalance (some attacks have <100 samples)
- Minimize false alarms to avoid alert fatigue
- Explain detections for analyst review
- Run fast enough for real-time deployment

## Solution

LightGBM classifier with intelligent sampling strategy and SHAP explainability.

## Results

**Test Set Performance:**
- Macro F1-Score: 0.980
- Macro Precision: 0.970
- Macro Recall: 1.000
- Accuracy: 0.988


## Technical Approach

**Pipeline:**
1. Preprocessing: Handle infinity values, impute missing data, memory optimization
2. Sampling: Undersample BENIGN (500K), oversample rare attacks (2-10% of majority)
3. Model: LightGBM (349 estimators, depth=5, lr=0.053, is_unbalance=True)
4. Validation: Monte Carlo cross-validation (multiple random splits)
5. Explainability: SHAP analysis for feature importance and attack signatures

**Key Decisions:**
- LightGBM over XGBoost: 10x faster, native imbalance support
- No ensemble stacking: Insufficient gain for added complexity
- Bot optimization: Favored recall (98%) over precision (69%) - catching attacks prioritized over false alarms

## Installation
```bash
pip install pandas numpy scikit-learn lightgbm imbalanced-learn shap matplotlib seaborn joblib
```

## Usage
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('NetworkIntrusionDetection.joblib')

# Predict on network flow data (80 features from CICIDS2017 format)
flow_data = pd.read_csv('your_flows.csv')
predictions = model.predict(flow_data)
probabilities = model.predict_proba(flow_data)
```

**Attack Signatures Learned:**
- **DDoS**: High packet rates, specific ports, elevated ACK/URG flags
- **DoS Slowloris**: Tiny segments, high idle time, low throughput
- **Bot**: Consistent IAT (low variance), fixed ports, stable up/down ratio
- **Heartbleed**: Abnormally large backward packets on port 443
- **Port Scan**: Rapid sequential connections, high PSH flags, diverse ports

The model learned protocol-level behavior, not superficial correlations.

## Model Architecture
```
Raw Data (2.8M flows, 80 features)
    ↓
Preprocessing (handle inf/NaN, optimize dtypes)
    ↓
Sampling (under+oversample to balance classes)
    ↓
LightGBM (349 trees, depth 5, tuned via Monte Carlo)
    ↓
Predictions + SHAP Explanations
```

## Dataset

CICIDS2017 - Canadian Institute for Cybersecurity Intrusion Detection System
- 2.8M network flows (5 days of realistic traffic)
- 10 attack categories: DDoS, DoS variants, PortScan, Brute Force, Web Attacks, Bot, Infiltration, Heartbleed, Benign
- 80+ features extracted from packet captures
- Source: https://www.unb.ca/cic/datasets/ids-2017.html

## Project Structure
```
network-intrusion-detection/
├── Datasets/combine.csv                    # CICIDS2017 data
├── network_intrusion/NetInt.ipynb          # Full analysis
├── NetworkIntrusionDetection.joblib        # Trained model
├── requirements.txt
└── README.md
```

## Future Work

- REST API deployment (FastAPI) for real-time inference
- Continuous model monitoring for concept drift
- Ensemble with deep learning for payload inspection
- Zero-day detection via anomaly detection

## Technologies

Python, pandas, scikit-learn, LightGBM, imbalanced-learn, SHAP, Matplotlib, Seaborn

## Contact

**Rafiou Diallo**  
Computer Science Student | ML + Security  
rafioudiallo12@gmail.com  
[GitHub](https://github.com/D-Rafiou) 
[LinkedIn](https://www.linkedin.com/in/rafiou-diallo-004522260/)
