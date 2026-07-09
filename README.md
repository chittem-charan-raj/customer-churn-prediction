# Customer Churn Prediction — Telco Dataset

An end-to-end ML pipeline predicting customer churn for a telecom provider, with a live interactive risk-scoring dashboard.

## Overview
This project trains and compares two classifiers on 7,043 customer records, explicitly weighing the interpretability-vs-accuracy trade-off rather than just chasing the top metric — a real deployment decision, not just a benchmark exercise.

## Results
| Model | ROC-AUC |
|---|---|
| Naive baseline ("predict everyone stays") | 73.5% accuracy |
| Logistic Regression | **84.2%** |
| Random Forest | 84.5% |

**Model selected for deployment: Logistic Regression.** Despite Random Forest's marginally higher AUC, Logistic Regression's coefficients can be read out directly ("flagged mainly due to contract type and tenure") — valuable when a churn flag needs to be explained to a customer or a compliance team.

## What's Included
- `churn_pipeline.py` — full pipeline: data loading, preprocessing, class-imbalance handling, model training, evaluation, and export of trained coefficients to JSON
- `model_params.json` — trained Logistic Regression coefficients (intercept, per-feature weights, scaling parameters) consumed by the dashboard
- `churn_app.html` — standalone interactive dashboard: enter a customer's attributes, get a real-time churn-risk score computed directly from the exported model coefficients
- `churn.csv` — source dataset (Telco Customer Churn, 7,043 records, 21 features)

## Visualizations
- `01_churn_baseline.png` — overall churn split: 73.5% stayed vs. 26.5% churned
- `02_churn_by_contract.png` — churn rate by contract type: 42.7% (month-to-month) vs. 11.3% (one year) vs. 2.8% (two year)
- `03_churn_by_tenure.png` — churn rate drops sharply with tenure: 47% in the first year down to ~7% after 5 years
- `04_roc_curve.png` — ROC comparison of both models (LogReg AUC 0.842 vs. Random Forest AUC 0.845)
- `05_probability_distribution.png` — predicted probability split by actual outcome, showing where the two classes overlap
- `06_calibration_curve.png` — checks whether a predicted 70% risk score actually corresponds to a ~70% real-world churn rate
- `07_confusion_matrix.png` — confusion matrix at a lowered decision threshold (0.3 instead of 0.5)
- `08_feature_importance.png` — standardized Logistic Regression coefficients, signed by whether they increase or decrease churn risk

## Tech Stack
- Python, scikit-learn, pandas, Matplotlib
- Vanilla HTML/CSS/JavaScript (dashboard — no backend required, model runs client-side from JSON)

## How to Run

**Train the model:**
```bash
pip install scikit-learn pandas matplotlib
python churn_pipeline.py
```

**Use the dashboard:**
Just open `churn_app.html` in a browser — it loads `model_params.json` and scores inputs entirely client-side.

## Key Insight
Churn rate in the raw data is 26.5% — meaningfully imbalanced. Class-weighted training was used rather than naive accuracy optimization, since a model that just predicts "no churn" for everyone would already be right 73.5% of the time while being completely useless for the actual business problem: identifying at-risk customers before they leave.

**Threshold tuning:** the default 0.5 cutoff misses too many actual churners for a business that wants to intervene early. Lowering the decision threshold to 0.3 trades precision for recall — catching 347 of 374 actual churners (missing only 27) at the cost of more false alarms (457 customers flagged who would have stayed). For a retention-campaign use case, that trade-off makes sense: the cost of a missed churner is usually higher than the cost of an unnecessary retention offer.

**Calibration matters, not just AUC:** the calibration curve shows the model is somewhat overconfident at high predicted probabilities — a customer scored at "0.7 risk" doesn't churn 70% of the time in practice. This is a distinction most churn projects skip entirely: a model can have strong ranking ability (high AUC) while still being poorly calibrated for probability-based decisions, and it's an important caveat to raise if this model were ever used to set actual intervention budgets.

**Strongest churn drivers** (from the standardized coefficients): short tenure, fiber-optic internet, and month-to-month contracts increase risk most; long tenure, no internet service, and two-year contracts are the strongest protective factors.
