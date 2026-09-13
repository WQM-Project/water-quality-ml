#  Water Quality Monitoring — Machine learning Models 

> ML pipeline for the RC boat Water quality monitoring project. Handles real-time water quality classification and anomaly detection for the River.

---


## Why Machine Learning Over Standard Formulas?

Traditional water quality assessment relies on the standard Water Quality Index (WQI) formula. While mathematically sound on paper, it completely fails in real-world, real-time autonomous deployments for two major reasons:

1. **The "Virtual Laboratory":** The standard WQI requires expensive, laboratory-grade parameters like BOD (Biochemical Oxygen Demand) and COD, which take 3–5 days to process. Our RC boat only carries low-cost electrochemical sensors. Machine Learning acts as a "virtual laboratory," learning the historical relationships between cheap sensor data and expensive lab results, allowing us to infer lab-grade quality instantly, right on the water.
2. **Complex River Ecosystems:** Mathematical formulas treat every parameter as a rigid, independent variable. In reality, rivers are complex. A rise in temperature changes how Dissolved Oxygen behaves; spikes in conductivity alter pH non-linearly. Machine Learning (specifically tree-based ensembles) inherently understands and maps these hidden, multidimensional relationships that standard WQI formulas completely miss.

---

## Quick Start

### 1. Setup
```bash
cd water-quality-ml
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Train Models (run in order)
```bash
python notebooks/01_eda.py              # Data exploration & cleaning
python notebooks/02_classification.py    # Train classifier
python notebooks/03_anomaly_detection.py # Train anomaly detector
python notebooks/04_forecasting.py       # Train forecaster (needs 'prophet')
```

---

## Feature Selection

An ablation study was conducted across 8 different feature combinations to determine the optimal sensor payload. The final model uses **7 independent features**:

- **pH** — Measures acidity/alkalinity. Irreplaceable indicator of chemical dumping.
- **Turbidity (NTU)** — Measures suspended solids via optical sensor. Critical for physical pollution detection.
- **Electrical Conductivity (µS/cm)** — Measures dissolved ion concentration. Most heavily weighted feature in the model.
- **Dissolved Oxygen (mg/L)** — Most critical metric for aquatic life survivability.
- **Chlorides (mg/L)** — Ablation study proved it contains unique chemical variance that improved accuracy by ~0.5%.
- **ORP (mV)** — Oxidation-Reduction Potential measures the water's ability to break down dead organic matter and contaminants. Highly sensitive to sudden influxes of biological pollution.
- **Temperature (°C)** — Influences chemical solubilities and biological activity. Provides crucial context for seasonal water quality variations.

**Dropped features:**
- **TDS** — Perfect linear derivative of Conductivity (`TDS = EC × 0.65`). Added exactly 0.000% information gain.
- **BOD & COD** — Used only to generate training labels. Dropped before model training to prevent data leakage.


---


### Model 1: Water Quality Classification (XGBoost)

An XGBoost classifier categorizes each sensor reading into one of four classes:

| Class | Label | Meaning |
|---|---|---|
| 0 | Excellent | Safe for all uses |
| 1 | Good | Minor treatment needed |
| 2 | Poor | Significant contamination |
| 3 | Dangerous | Immediate ecological threat |

The model was trained on an 80/20 stratified split with balanced class weights to handle rare "Dangerous" samples. Inference takes less than 1 millisecond per reading.

---

### Model 2: Two-Layer Anomaly Detection

A hybrid system designed to catch both known and unknown pollution events:

**Layer 1 — Rule-Based Engine:**
Hard scientific thresholds from CPCB/BIS standards. If any sensor crosses a known danger limit (e.g., pH < 4.0 or DO < 1.0 mg/L), the alarm triggers instantly without consulting the AI. This layer is deterministic and cannot fail.

**Layer 2 — Isolation Forest (Unsupervised ML):**
If all rules pass, the AI checks whether the *combination* of sensor values is physically plausible for a natural river. The Isolation Forest algorithm learns the multi-dimensional shape of normal river water and flags readings that are statistically easy to isolate from the crowd. This catches unprecedented events like an invisible chemical solvent (normal pH, normal turbidity, but abnormally high conductivity) that no single rule can detect.

---

### Upcoming: Time Series Forecasting (Prophet)

The next phase of the ML pipeline will introduce **Facebook Prophet** for time series forecasting. The goal is to predict water quality trends 24–72 hours into the future based on historical seasonal patterns. This will enable:

- **Early Warning System** — Alert authorities *before* water quality deteriorates, not after.
- **Seasonal Trend Analysis** — Identify recurring pollution patterns tied to monsoon runoff, agricultural cycles, or industrial discharge schedules.
- **Preventive Action** — Shift from reactive monitoring ("the water is bad right now") to proactive forecasting ("the water will likely degrade by Thursday").

---

### How to Run

```bash
# Step 1: Clean and prepare the dataset
python notebooks/02_beas_eda.py

# Step 2: Train the XGBoost classification model
python notebooks/03_beas_classification.py

# Step 3: Train the Two-Layer anomaly detection system
python notebooks/04_beas_anomaly_detection.py
