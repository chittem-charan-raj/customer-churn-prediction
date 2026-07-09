"""
Customer Churn Prediction — Business-Framed Logistic Regression + PNG Reports
================================================================================

The two skills this project is actually about:

1. BUSINESS ML THINKING — accuracy alone is misleading here. ~73% of
   customers don't churn, so a model that predicts "nobody churns" gets
   73% accuracy while being completely useless. The real questions are:
   does this beat that naive baseline, and what does a false negative
   (missed at-risk customer) cost compared to a false positive (an
   unnecessary retention offer)? That cost asymmetry is what should set
   your decision threshold — not a default of 0.5.

2. PROBABILITY OUTPUT INTERPRETATION — predict_proba() returns a
   continuous risk score, not a label. A 0.35 and a 0.85 are both
   "predicted to stay" under a 0.5 cutoff, but they are very different
   customers from a retention-team's point of view. This script keeps
   the probability visible end-to-end instead of collapsing it early.

Dataset: the standard "Telco Customer Churn" schema (7043 rows, the same
one used across most churn tutorials and the Kaggle dataset by blastchar).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "churn.csv")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 — Load raw data and confirm the baseline
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Raw shape:", df.shape)

churn_rate = (df["Churn"] == "Yes").mean()
print(f"\nOverall churn rate: {churn_rate:.1%}")
print(f"Naive 'predict everyone stays' baseline accuracy: {1 - churn_rate:.1%}")
print("^ any model we build has to clearly beat this, not just have 'high accuracy'.")

plt.figure(figsize=(5, 4))
plt.bar(["Stayed", "Churned"], [1 - churn_rate, churn_rate], color=["#4C72B0", "#C44E52"])
plt.title("Customer Base: Churn Rate")
plt.ylabel("Share of customers")
for i, v in enumerate([1 - churn_rate, churn_rate]):
    plt.text(i, v + 0.01, f"{v:.1%}", ha="center")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "01_churn_baseline.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# STEP 2 — A couple of quick, business-relevant EDA charts (full data is
# fine here — these are descriptive only, nothing is fit/learned from them)
# ---------------------------------------------------------------------------
contract_churn = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean())
contract_churn = contract_churn.reindex(["Month-to-month", "One year", "Two year"])
plt.figure(figsize=(6, 4))
plt.bar(contract_churn.index, contract_churn.values, color="#DD8452")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn rate")
for i, v in enumerate(contract_churn.values):
    plt.text(i, v + 0.01, f"{v:.1%}", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "02_churn_by_contract.png"), dpi=150)
plt.close()

tenure_bins = [0, 12, 24, 48, 60, 72]
tenure_labels = ["0-12 mo", "13-24 mo", "25-48 mo", "49-60 mo", "61-72 mo"]
df["_TenureGroup"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True)
tenure_churn = df.groupby("_TenureGroup")["Churn"].apply(lambda s: (s == "Yes").mean())
plt.figure(figsize=(6, 4))
plt.bar(tenure_churn.index.astype(str), tenure_churn.values, color="#55A868")
plt.title("Churn Rate by Tenure (New Customers Are the Risk)")
plt.ylabel("Churn rate")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "03_churn_by_tenure.png"), dpi=150)
plt.close()
df = df.drop(columns=["_TenureGroup"])


# ---------------------------------------------------------------------------
# STEP 3 — Split FIRST, clean SECOND (same leak-safe rule as always)
# ---------------------------------------------------------------------------
df["Churn"] = (df["Churn"] == "Yes").astype(int)
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ---------------------------------------------------------------------------
# STEP 4 — Feature engineering / cleaning transformer
# ---------------------------------------------------------------------------
# Three real data-quality issues in this dataset, each handled deliberately:
#   - TotalCharges is stored as TEXT and has 11 blank strings (not NaN) —
#     all from customers with tenure=0 who simply haven't been billed yet.
#     We fill those with 0, using that domain fact, not a generic median.
#   - Six "service add-on" columns use THREE category values each
#     ("Yes" / "No" / "No internet service") where the third value is
#     just a redundant restatement of InternetService=="No" — we collapse
#     it to "No" so the model doesn't spend a parameter relearning that.
#   - We add NumAddOnServices: how many of those six add-ons a customer
#     has, since "bundled in" customers churn measurably less.
class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                  "TechSupport", "StreamingTV", "StreamingMovies"]

    def fit(self, X, y=None):
        return self  # rule-based only — nothing learned, safe pre-split or post

    def transform(self, X):
        X = X.copy()

        # TotalCharges: text -> numeric, domain-informed fill for new customers
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X.loc[X["TotalCharges"].isna() & (X["tenure"] == 0), "TotalCharges"] = 0.0

        # Collapse redundant placeholder categories
        for col in self.ADDON_COLS + ["MultipleLines"]:
            X[col] = X[col].replace({"No internet service": "No", "No phone service": "No"})

        # Bundle-depth signal
        X["NumAddOnServices"] = (X[self.ADDON_COLS] == "Yes").sum(axis=1)

        return X.drop(columns=["customerID"], errors="ignore")


# ---------------------------------------------------------------------------
# STEP 5 — Encoding via ColumnTransformer (fit on train only)
# ---------------------------------------------------------------------------
# Logistic Regression — unlike the Random Forest in the Titanic project —
# DOES need scaled numeric inputs: its coefficients are only comparable to
# each other, and its solver converges reliably, when features share a
# common scale. drop="if_binary" keeps genuinely two-valued columns (like
# Partner: Yes/No) as a single 0/1 column instead of two redundant ones,
# which keeps the coefficient table clean and directly interpretable.
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddOnServices", "SeniorCitizen"]
categorical_features = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

numeric_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])


# ---------------------------------------------------------------------------
# STEP 6 — Two models on identical preprocessing: interpretable vs. accurate
# ---------------------------------------------------------------------------
# class_weight="balanced" matters here: with a 73/27 split, an unweighted
# model can quietly lean toward always predicting "stays" and still look
# decent on raw accuracy. Balancing reweights the minority (churn) class
# during training so it actually has to learn to recognize churners.
logreg_pipeline = Pipeline(steps=[
    ("feature_engineering", ChurnFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])

rf_pipeline = Pipeline(steps=[
    ("feature_engineering", ChurnFeatureEngineer()),
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])

logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

logreg_auc = roc_auc_score(y_test, logreg_proba)
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"\nLogistic Regression ROC-AUC: {logreg_auc:.3f}")
print(f"Random Forest ROC-AUC:       {rf_auc:.3f}")
print(
    "\nThe business call: even if Random Forest's AUC is a bit higher, "
    "Logistic Regression's coefficients can be read out and explained to a "
    "compliance team or a customer ('flagged mainly due to contract type and "
    "tenure') — that explainability is often worth more than a small accuracy "
    "gain, especially in regulated industries like banking. We deploy LogReg."
)

fpr_lr, tpr_lr, _ = roc_curve(y_test, logreg_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={logreg_auc:.3f})", color="#4C72B0")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})", color="#DD8452")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "04_roc_curve.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# STEP 7 — Probability interpretation: distribution + calibration
# ---------------------------------------------------------------------------
# This is the core "read probabilities, not labels" visual. A well-separated
# model pushes churners toward 1.0 and stayers toward 0.0; heavy overlap in
# the middle means the model genuinely isn't sure about a lot of customers
# — which itself is useful business information (that's your "monitor
# closely" tier, distinct from "definitely fine" or "act now").
plt.figure(figsize=(7, 4))
plt.hist(logreg_proba[y_test.values == 0], bins=30, alpha=0.6, label="Actually stayed", color="#4C72B0")
plt.hist(logreg_proba[y_test.values == 1], bins=30, alpha=0.6, label="Actually churned", color="#C44E52")
plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Default 0.5 cutoff")
plt.xlabel("Predicted churn probability")
plt.ylabel("Customer count")
plt.title("Predicted Probability Distribution by Actual Outcome")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "05_probability_distribution.png"), dpi=150)
plt.close()

frac_pos, mean_pred = calibration_curve(y_test, logreg_proba, n_bins=10)
plt.figure(figsize=(5.5, 5))
plt.plot(mean_pred, frac_pos, marker="o", color="#4C72B0", label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability (per bin)")
plt.ylabel("Actual churn rate (per bin)")
plt.title("Calibration: Does a 0.7 Really Mean ~70% Risk?")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "06_calibration_curve.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# STEP 8 — Threshold is a BUSINESS decision, not a default
# ---------------------------------------------------------------------------
# A missed churner (false negative) costs a customer's remaining lifetime
# value — often hundreds of dollars. A false positive just costs a retention
# email or a small discount offered to someone who'd have stayed anyway —
# a few dollars. That asymmetry justifies a LOWER threshold than 0.5: we'd
# rather over-flag and occasionally waste a discount than under-flag and
# lose a customer outright. 0.3 is a common starting point in practice;
# the "right" number depends on your actual retention-offer cost vs.
# average customer lifetime value.
print("\nThreshold comparison (this is a business choice, not a fixed rule):")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Flagged':>10}")
for threshold in [0.3, 0.5, 0.7]:
    preds = (logreg_proba >= threshold).astype(int)
    p = precision_score(y_test, preds)
    r = recall_score(y_test, preds)
    flagged = preds.sum()
    print(f"{threshold:>10.1f} {p:>10.2f} {r:>10.2f} {flagged:>10d}")

BUSINESS_THRESHOLD = 0.3
final_preds = (logreg_proba >= BUSINESS_THRESHOLD).astype(int)
print(f"\nUsing business threshold = {BUSINESS_THRESHOLD} for the deployed report below:")
print(classification_report(y_test, final_preds, target_names=["Stayed", "Churned"]))

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Reds")
plt.title(f"Confusion Matrix (threshold = {BUSINESS_THRESHOLD})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
labels = ["Stayed", "Churned"]
plt.xticks([0, 1], labels)
plt.yticks([0, 1], labels)
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "07_confusion_matrix.png"), dpi=150)
plt.close()


# ---------------------------------------------------------------------------
# STEP 9 — Coefficients: which factors actually drive the prediction
# ---------------------------------------------------------------------------
ohe = preprocessor.named_transformers_["cat"].named_steps["encode"]
cat_names = ohe.get_feature_names_out(categorical_features)
all_names = np.concatenate([numeric_features, cat_names])
coefs = logreg_pipeline.named_steps["classifier"].coef_[0]

coef_df = (
    pd.DataFrame({"feature": all_names, "coefficient": coefs})
    .sort_values("coefficient", key=abs, ascending=False)
    .head(12)
)
print("\nTop 12 features by |coefficient| (positive = raises churn risk):")
print(coef_df.to_string(index=False))

colors = ["#C44E52" if c > 0 else "#4C72B0" for c in coef_df["coefficient"][::-1]]
plt.figure(figsize=(7, 6))
plt.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression Coefficients\n(red = increases churn risk, blue = decreases it)")
plt.xlabel("Coefficient (standardized scale)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "08_feature_importance.png"), dpi=150)
plt.close()

print(f"\nSaved 8 plots to: {PLOTS_DIR}")


# ---------------------------------------------------------------------------
# STEP 10 — Refit on ALL data for the deployed model
# ---------------------------------------------------------------------------
# Steps 1-9 used an 80/20 split so the AUC/precision/recall numbers above are
# an honest, unbiased estimate of real-world performance. Now that we trust
# those numbers, refit the same pipeline on every row we have — throwing
# away 20% of your data in the model that actually goes into production has
# no upside once validation is done.
deploy_pipeline = Pipeline(steps=[
    ("feature_engineering", ChurnFeatureEngineer()),
    ("preprocessor", ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))]), categorical_features),
    ])),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
deploy_pipeline.fit(X, y)


# ---------------------------------------------------------------------------
# STEP 11 — Export exact model parameters for the standalone web UI
# ---------------------------------------------------------------------------
# The UI (churn_app.html) is a single self-contained file — no Python server
# behind it — so it needs the EXACT numbers the model uses: scaler means/
# scales and logistic regression coefficients, structured so JavaScript can
# rebuild the identical feature vector and run the identical sigmoid.
deploy_scaler = deploy_pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["scale"]
deploy_ohe = deploy_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encode"]
deploy_coefs = deploy_pipeline.named_steps["classifier"].coef_[0]
deploy_intercept = float(deploy_pipeline.named_steps["classifier"].intercept_[0])

ENGINEERED_NUMERIC = {"NumAddOnServices"}  # computed from toggles, not a direct user input
ADDON_COLS = ChurnFeatureEngineer.ADDON_COLS

numeric_export = []
for i, feat in enumerate(numeric_features):
    entry = {
        "name": feat,
        "mean": float(deploy_scaler.mean_[i]),
        "scale": float(deploy_scaler.scale_[i]),
        "coef": float(deploy_coefs[i]),
    }
    if feat in ENGINEERED_NUMERIC:
        entry["source_toggles"] = ADDON_COLS
    numeric_export.append(entry)

categorical_export = []
coef_offset = len(numeric_features)
for i, feat in enumerate(categorical_features):
    cats = list(deploy_ohe.categories_[i])
    drop_idx = deploy_ohe.drop_idx_[i] if deploy_ohe.drop_idx_ is not None else None
    if drop_idx is not None:
        cats.pop(drop_idx)
    coefs_for_col = {cat: float(deploy_coefs[coef_offset + j]) for j, cat in enumerate(cats)}
    all_cats = list(deploy_ohe.categories_[i])  # full option list for the dropdown, including the dropped/reference one
    categorical_export.append({"name": feat, "categories": all_cats, "coefs": coefs_for_col})
    coef_offset += len(cats)

model_export = {
    "intercept": deploy_intercept,
    "numeric": numeric_export,
    "categorical": categorical_export,
    "addon_cols": ADDON_COLS,
}

import json
with open(os.path.join(SCRIPT_DIR, "model_params.json"), "w") as f:
    json.dump(model_export, f, indent=2)
print(f"\nExported deployment model parameters to: model_params.json")


# ---------------------------------------------------------------------------
# STEP 12 — Verify the export is exact, not approximate
# ---------------------------------------------------------------------------
# Re-derive predicted probability by hand from model_export alone, for a
# handful of real customers, and compare against deploy_pipeline.predict_proba.
# This is the same arithmetic churn_app.html's JavaScript will perform — if it
# matches here, the HTML version is correct by construction, not by hope.
def manual_predict_proba(row, params):
    logit = params["intercept"]
    for nf in params["numeric"]:
        if "source_toggles" in nf:
            value = sum(1 for c in nf["source_toggles"] if row[c] == "Yes")
        else:
            value = row[nf["name"]]
        scaled = (value - nf["mean"]) / nf["scale"]
        logit += scaled * nf["coef"]
    for cf in params["categorical"]:
        val = row[cf["name"]]
        if val in {"No internet service", "No phone service"}:
            val = "No"  # mirror ChurnFeatureEngineer's collapse rule
        logit += cf["coefs"].get(val, 0.0)
    return 1 / (1 + np.exp(-logit))

check_rows = X.sample(5, random_state=7).copy()
check_rows["TotalCharges"] = pd.to_numeric(check_rows["TotalCharges"], errors="coerce").fillna(0.0)
sklearn_probs = deploy_pipeline.predict_proba(check_rows)[:, 1]

print("\nVerifying hand-rolled probability math against sklearn (must match closely):")
print(f"{'sklearn':>10} {'manual':>10} {'abs diff':>10}")
max_diff = 0.0
for (idx, row), sk_p in zip(check_rows.iterrows(), sklearn_probs):
    manual_p = manual_predict_proba(row, model_export)
    diff = abs(sk_p - manual_p)
    max_diff = max(max_diff, diff)
    print(f"{sk_p:>10.4f} {manual_p:>10.4f} {diff:>10.6f}")

if max_diff < 1e-6:
    print("MATCH CONFIRMED — safe to port this exact arithmetic into JavaScript.")
else:
    print(f"MISMATCH (max diff {max_diff:.6f}) — do not build the UI on this export yet.")


# ---------------------------------------------------------------------------
# STEP 13 — Generate the standalone web UI with real parameters baked in
# ---------------------------------------------------------------------------
# churn_app.html is fully self-contained: open it by double-clicking, no
# Python, no server, no internet required for it to compute predictions
# (it does load Google Fonts over the network for typography only - the
# risk math works fully offline either way). The placeholder below is
# swapped for the verified model_export JSON from Step 11.
HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ChurnGuard — Customer Risk Assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#F3F5F7;
    --surface:#FFFFFF;
    --border:#E1E6EA;
    --ink:#16232E;
    --muted:#5B6B79;
    --accent:#0E6E64;
    --accent-soft:#E3F0EE;
    --low:#1E7F4F;
    --low-soft:#E7F5ED;
    --medium:#B6740B;
    --medium-soft:#FBF0DD;
    --high:#B23A33;
    --high-soft:#FBEAE8;
    --radius:8px;
  }
  *{box-sizing:border-box;}
  html,body{margin:0;padding:0;}
  body{
    background:var(--bg);
    color:var(--ink);
    font-family:'IBM Plex Sans', -apple-system, Segoe UI, sans-serif;
    font-size:14px;
    line-height:1.45;
    -webkit-font-smoothing:antialiased;
  }
  .mono{font-family:'IBM Plex Mono', monospace;}

  /* ---------- top bar ---------- */
  .topbar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:14px 28px;
    background:var(--surface);
    border-bottom:1px solid var(--border);
  }
  .brand{display:flex;align-items:center;gap:10px;}
  .brand-mark{
    width:30px;height:30px;border-radius:7px;
    background:linear-gradient(135deg, var(--accent), #0A4A43);
    display:flex;align-items:center;justify-content:center;
    color:#fff;font-weight:700;font-size:14px;font-family:'IBM Plex Mono',monospace;
    flex-shrink:0;
  }
  .brand-text{display:flex;flex-direction:column;line-height:1.15;}
  .brand-name{font-weight:700;font-size:15px;letter-spacing:.2px;}
  .brand-sub{font-size:11.5px;color:var(--muted);}
  .context-tag{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;font-weight:600;letter-spacing:.06em;
    color:var(--accent);background:var(--accent-soft);
    padding:5px 10px;border-radius:5px;
  }

  /* ---------- layout ---------- */
  .layout{
    max-width:1180px;margin:0 auto;
    display:grid;grid-template-columns:1.35fr 1fr;gap:20px;
    padding:24px 28px 60px;
    align-items:start;
  }
  @media (max-width: 880px){
    .layout{grid-template-columns:1fr;}
  }

  .card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:18px 20px 20px;
    margin-bottom:16px;
  }
  .card-eyebrow{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;font-weight:600;letter-spacing:.08em;
    color:var(--muted);text-transform:uppercase;
    margin:0 0 4px;
  }
  .card-title{font-size:16px;font-weight:600;margin:0 0 14px;}

  .field-grid{
    display:grid;grid-template-columns:1fr 1fr;gap:14px 16px;
  }
  .field{display:flex;flex-direction:column;gap:5px;}
  .field.span2{grid-column:1 / -1;}
  .field label{font-size:12.5px;color:var(--muted);font-weight:500;}
  .field input[type=number], .field select{
    font-family:inherit;font-size:13.5px;color:var(--ink);
    border:1px solid var(--border);border-radius:6px;
    padding:8px 10px;background:#fff;
    transition:border-color .15s;
  }
  .field input[type=number]:focus, .field select:focus{
    outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft);
  }
  .readonly-display{
    font-family:'IBM Plex Mono',monospace;font-size:13.5px;
    padding:8px 10px;background:#F7F9FA;border:1px solid var(--border);
    border-radius:6px;color:var(--muted);
  }

  /* toggle switch */
  .toggle-row{
    display:flex;align-items:center;justify-content:space-between;
    padding:4px 0;
  }
  .toggle-row label{font-size:13px;color:var(--ink);font-weight:500;}
  .toggle-row .hint{font-size:11px;color:var(--muted);font-weight:400;display:block;}
  .switch{position:relative;width:42px;height:24px;flex-shrink:0;}
  .switch input{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer;z-index:2;}
  .switch .track{
    position:absolute;inset:0;background:#D5DBE0;border-radius:999px;
    transition:background .15s;
  }
  .switch .thumb{
    position:absolute;top:3px;left:3px;width:18px;height:18px;border-radius:50%;
    background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);
    transition:transform .15s;
  }
  .switch input:checked + .track{background:var(--accent);}
  .switch input:checked + .track + .thumb{transform:translateX(18px);}
  .switch input:focus-visible + .track{box-shadow:0 0 0 3px var(--accent-soft);}
  .field-disabled{opacity:.45;}
  .field-disabled .switch input{cursor:not-allowed;}

  .divider-label{
    font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--muted);
    text-transform:uppercase;letter-spacing:.07em;
    margin:14px 0 8px;display:flex;align-items:center;gap:8px;
  }
  .divider-label::after{content:"";flex:1;height:1px;background:var(--border);}

  /* ---------- risk column ---------- */
  .risk-col{position:sticky;top:24px;}
  .risk-card{
    background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:24px 22px;text-align:center;margin-bottom:16px;
  }
  .risk-score{
    font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:52px;
    line-height:1;transition:color .25s;
  }
  .risk-tier{
    display:inline-block;margin-top:8px;padding:4px 12px;border-radius:999px;
    font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;
    transition:background .25s, color .25s;
  }
  .risk-meter{
    margin-top:16px;height:8px;border-radius:999px;background:#E9ECEF;position:relative;overflow:hidden;
  }
  .risk-meter-fill{height:100%;border-radius:999px;transition:width .25s, background .25s;}

  .factors-card{
    background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
    padding:18px 20px;margin-bottom:16px;
  }
  .factors-card h3{font-size:13px;font-weight:600;margin:0 0 4px;}
  .factors-sub{font-size:11.5px;color:var(--muted);margin:0 0 14px;}
  .factor-row{display:grid;grid-template-columns:120px 1fr;align-items:center;gap:10px;margin-bottom:9px;}
  .factor-label{font-size:11.5px;color:var(--ink);text-align:right;line-height:1.25;}
  .factor-track{position:relative;height:16px;}
  .factor-track::before{
    content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--border);
  }
  .factor-bar{
    position:absolute;top:3px;height:10px;border-radius:3px;
    transition:width .25s, left .25s;
  }
  .factor-bar.risk{background:var(--high);}
  .factor-bar.protect{background:var(--low);}

  .action-card{
    border-radius:var(--radius);padding:16px 18px;border:1px solid var(--border);
    transition:background .25s, border-color .25s;
  }
  .action-card h3{font-size:12px;font-weight:600;margin:0 0 6px;text-transform:uppercase;letter-spacing:.04em;}
  .action-card p{margin:0;font-size:13px;line-height:1.5;}

  .footnote{font-size:11px;color:var(--muted);margin-top:14px;text-align:center;}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <div class="brand-mark">CG</div>
    <div class="brand-text">
      <span class="brand-name">ChurnGuard</span>
      <span class="brand-sub">Customer Risk Assessment</span>
    </div>
  </div>
  <div class="context-tag">TELECOM RETENTION DESK</div>
</div>

<div class="layout">
  <!-- ===================== FORM COLUMN ===================== -->
  <section class="form-col">

    <div class="card">
      <p class="card-eyebrow">Account</p>
      <h2 class="card-title">Plan &amp; billing</h2>
      <div class="field-grid">
        <div class="field">
          <label for="tenure">Tenure (months)</label>
          <input type="number" id="tenure" min="0" max="72" value="8" />
        </div>
        <div class="field">
          <label for="MonthlyCharges">Monthly charges ($)</label>
          <input type="number" id="MonthlyCharges" min="0" max="200" step="0.01" value="84.50" />
        </div>
        <div class="field">
          <label for="Contract">Contract</label>
          <select id="Contract">
            <option value="Month-to-month">Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>
        <div class="field">
          <label for="PaymentMethod">Payment method</label>
          <select id="PaymentMethod">
            <option value="Electronic check">Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option>
          </select>
        </div>
        <div class="field span2">
          <label>Total billed to date</label>
          <div class="readonly-display" id="TotalChargesDisplay">$676.00</div>
        </div>
      </div>
      <div class="divider-label">Billing preference</div>
      <div class="toggle-row">
        <label for="PaperlessBilling">Paperless billing</label>
        <span class="switch">
          <input type="checkbox" id="PaperlessBilling" />
          <span class="track"></span><span class="thumb"></span>
        </span>
      </div>
    </div>

    <div class="card">
      <p class="card-eyebrow">Services</p>
      <h2 class="card-title">Subscribed services</h2>

      <div class="toggle-row">
        <label for="PhoneService">Phone service</label>
        <span class="switch">
          <input type="checkbox" id="PhoneService" checked />
          <span class="track"></span><span class="thumb"></span>
        </span>
      </div>
      <div class="toggle-row" id="row-MultipleLines">
        <label for="MultipleLines">Multiple lines</label>
        <span class="switch">
          <input type="checkbox" id="MultipleLines" />
          <span class="track"></span><span class="thumb"></span>
        </span>
      </div>

      <div class="divider-label">Internet</div>
      <div class="field" style="margin-bottom:10px;">
        <label for="InternetService">Internet service</label>
        <select id="InternetService">
          <option value="DSL">DSL</option>
          <option value="Fiber optic" selected>Fiber optic</option>
          <option value="No">No internet service</option>
        </select>
      </div>
      <div id="addon-toggles">
        <div class="toggle-row" data-addon>
          <label for="OnlineSecurity">Online security</label>
          <span class="switch"><input type="checkbox" id="OnlineSecurity" /><span class="track"></span><span class="thumb"></span></span>
        </div>
        <div class="toggle-row" data-addon>
          <label for="OnlineBackup">Online backup</label>
          <span class="switch"><input type="checkbox" id="OnlineBackup" /><span class="track"></span><span class="thumb"></span></span>
        </div>
        <div class="toggle-row" data-addon>
          <label for="DeviceProtection">Device protection</label>
          <span class="switch"><input type="checkbox" id="DeviceProtection" /><span class="track"></span><span class="thumb"></span></span>
        </div>
        <div class="toggle-row" data-addon>
          <label for="TechSupport">Tech support</label>
          <span class="switch"><input type="checkbox" id="TechSupport" /><span class="track"></span><span class="thumb"></span></span>
        </div>
        <div class="toggle-row" data-addon>
          <label for="StreamingTV">Streaming TV</label>
          <span class="switch"><input type="checkbox" id="StreamingTV" /><span class="track"></span><span class="thumb"></span></span>
        </div>
        <div class="toggle-row" data-addon>
          <label for="StreamingMovies">Streaming movies</label>
          <span class="switch"><input type="checkbox" id="StreamingMovies" /><span class="track"></span><span class="thumb"></span></span>
        </div>
      </div>
    </div>

    <div class="card">
      <p class="card-eyebrow">Demographics</p>
      <h2 class="card-title">Household</h2>
      <div class="toggle-row">
        <label for="gender">Male</label>
        <span class="switch"><input type="checkbox" id="gender" /><span class="track"></span><span class="thumb"></span></span>
      </div>
      <div class="toggle-row">
        <label for="SeniorCitizen">Senior citizen (65+)</label>
        <span class="switch"><input type="checkbox" id="SeniorCitizen" /><span class="track"></span><span class="thumb"></span></span>
      </div>
      <div class="toggle-row">
        <label for="Partner">Has partner</label>
        <span class="switch"><input type="checkbox" id="Partner" /><span class="track"></span><span class="thumb"></span></span>
      </div>
      <div class="toggle-row">
        <label for="Dependents">Has dependents</label>
        <span class="switch"><input type="checkbox" id="Dependents" /><span class="track"></span><span class="thumb"></span></span>
      </div>
    </div>

  </section>

  <!-- ===================== RISK COLUMN ===================== -->
  <aside class="risk-col">
    <div class="risk-card">
      <div class="risk-score mono" id="riskScore">--%</div>
      <div class="risk-tier" id="riskTier">—</div>
      <div class="risk-meter"><div class="risk-meter-fill" id="riskMeterFill"></div></div>
    </div>

    <div class="factors-card">
      <h3>Top factors for this customer</h3>
      <p class="factors-sub">Red pushes risk up · green pulls it down</p>
      <div id="factorsList"></div>
    </div>

    <div class="action-card" id="actionCard">
      <h3>Recommended action</h3>
      <p id="actionText">—</p>
    </div>

    <p class="footnote">Probabilities from a logistic regression trained on historical churn outcomes. A model output, not a guarantee — use judgment.</p>
  </aside>
</div>

<script>
/* MODEL_PARAMS is injected by churn_pipeline.py at generation time. */
const MODEL_PARAMS = /*__MODEL_PARAMS__*/{"intercept":0,"numeric":[],"categorical":[],"addon_cols":[]}/*__END_MODEL_PARAMS__*/;

function predictChurnProbability(customer, params) {
  let logit = params.intercept;
  for (const nf of params.numeric) {
    let value;
    if (nf.source_toggles) {
      value = nf.source_toggles.filter((c) => customer[c] === "Yes").length;
    } else {
      value = customer[nf.name];
    }
    const scaled = (value - nf.mean) / nf.scale;
    logit += scaled * nf.coef;
  }
  for (const cf of params.categorical) {
    let val = customer[cf.name];
    if (val === "No internet service" || val === "No phone service") val = "No";
    logit += cf.coefs[val] || 0;
  }
  return 1 / (1 + Math.exp(-logit));
}

function contributions(customer, params) {
  const items = [];
  for (const nf of params.numeric) {
    let value;
    if (nf.source_toggles) value = nf.source_toggles.filter((c) => customer[c] === "Yes").length;
    else value = customer[nf.name];
    const scaled = (value - nf.mean) / nf.scale;
    items.push({ name: nf.name, value, contribution: scaled * nf.coef });
  }
  for (const cf of params.categorical) {
    let val = customer[cf.name];
    if (val === "No internet service" || val === "No phone service") val = "No";
    items.push({ name: cf.name, value: customer[cf.name], contribution: cf.coefs[val] || 0 });
  }
  return items;
}

const LABELS = {
  tenure: (v) => `Tenure: ${v} mo`,
  MonthlyCharges: (v) => `Monthly: $${Number(v).toFixed(0)}`,
  TotalCharges: (v) => `Total billed: $${Number(v).toFixed(0)}`,
  NumAddOnServices: (v) => `Add-ons: ${v} of 6`,
  SeniorCitizen: (v) => (v ? "Senior citizen" : "Not a senior"),
  gender: (v) => (v === "Male" ? "Gender: Male" : "Gender: Female"),
  Partner: (v) => (v === "Yes" ? "Has partner" : "No partner"),
  Dependents: (v) => (v === "Yes" ? "Has dependents" : "No dependents"),
  PhoneService: (v) => (v === "Yes" ? "Has phone service" : "No phone service"),
  MultipleLines: (v) => (v === "Yes" ? "Multiple phone lines" : "Single phone line"),
  InternetService: (v) => `Internet: ${v === "No" ? "none" : v}`,
  OnlineSecurity: (v) => (v === "Yes" ? "Has online security" : "No online security"),
  OnlineBackup: (v) => (v === "Yes" ? "Has online backup" : "No online backup"),
  DeviceProtection: (v) => (v === "Yes" ? "Has device protection" : "No device protection"),
  TechSupport: (v) => (v === "Yes" ? "Has tech support" : "No tech support"),
  StreamingTV: (v) => (v === "Yes" ? "Streams TV" : "No TV streaming"),
  StreamingMovies: (v) => (v === "Yes" ? "Streams movies" : "No movie streaming"),
  Contract: (v) => `Contract: ${v}`,
  PaperlessBilling: (v) => (v === "Yes" ? "Paperless billing" : "Paper billing"),
  PaymentMethod: (v) => `Pays via: ${v}`,
};

function readCustomer() {
  const phoneOn = document.getElementById("PhoneService").checked;
  const internet = document.getElementById("InternetService").value;
  const internetOn = internet !== "No";

  // conditional disabling mirrors real data dependencies
  document.getElementById("row-MultipleLines").classList.toggle("field-disabled", !phoneOn);
  document.getElementById("MultipleLines").disabled = !phoneOn;
  document.querySelectorAll("[data-addon]").forEach((row) => {
    row.classList.toggle("field-disabled", !internetOn);
    row.querySelector("input").disabled = !internetOn;
  });

  const tenure = Number(document.getElementById("tenure").value) || 0;
  const monthly = Number(document.getElementById("MonthlyCharges").value) || 0;
  const totalCharges = tenure * monthly;
  document.getElementById("TotalChargesDisplay").textContent = `$${totalCharges.toFixed(2)}`;

  const yn = (id) => (document.getElementById(id).checked ? "Yes" : "No");

  return {
    tenure,
    MonthlyCharges: monthly,
    TotalCharges: totalCharges,
    SeniorCitizen: document.getElementById("SeniorCitizen").checked ? 1 : 0,
    gender: document.getElementById("gender").checked ? "Male" : "Female",
    Partner: yn("Partner"),
    Dependents: yn("Dependents"),
    PhoneService: phoneOn ? "Yes" : "No",
    MultipleLines: phoneOn ? yn("MultipleLines") : "No",
    InternetService: internet,
    OnlineSecurity: internetOn ? yn("OnlineSecurity") : "No",
    OnlineBackup: internetOn ? yn("OnlineBackup") : "No",
    DeviceProtection: internetOn ? yn("DeviceProtection") : "No",
    TechSupport: internetOn ? yn("TechSupport") : "No",
    StreamingTV: internetOn ? yn("StreamingTV") : "No",
    StreamingMovies: internetOn ? yn("StreamingMovies") : "No",
    Contract: document.getElementById("Contract").value,
    PaperlessBilling: yn("PaperlessBilling"),
    PaymentMethod: document.getElementById("PaymentMethod").value,
  };
}

function tierFor(p) {
  if (p < 0.3) return { key: "low", label: "Low risk", color: "var(--low)", soft: "var(--low-soft)" };
  if (p < 0.6) return { key: "medium", label: "Medium risk", color: "var(--medium)", soft: "var(--medium-soft)" };
  return { key: "high", label: "High risk", color: "var(--high)", soft: "var(--high-soft)" };
}

const ACTIONS = {
  low: "No action needed. Account looks stable under current terms — standard service is fine.",
  medium: "Worth a proactive touch. A loyalty check-in or a soft mention of contract options at the next contact can head off drift.",
  high: "Flag for retention outreach now. Lead with the contract or service factor below — a targeted offer addressing it outperforms a generic discount.",
};

function render() {
  const customer = readCustomer();
  const p = predictChurnProbability(customer, MODEL_PARAMS);
  const tier = tierFor(p);

  const scoreEl = document.getElementById("riskScore");
  scoreEl.textContent = `${Math.round(p * 100)}%`;
  scoreEl.style.color = tier.color;

  const tierEl = document.getElementById("riskTier");
  tierEl.textContent = tier.label.toUpperCase();
  tierEl.style.background = tier.soft;
  tierEl.style.color = tier.color;

  const fill = document.getElementById("riskMeterFill");
  fill.style.width = `${Math.max(4, p * 100)}%`;
  fill.style.background = tier.color;

  const actionCard = document.getElementById("actionCard");
  actionCard.style.background = tier.soft;
  actionCard.style.borderColor = tier.color;
  document.getElementById("actionText").textContent = ACTIONS[tier.key];

  const items = contributions(customer, MODEL_PARAMS)
    .filter((it) => Math.abs(it.contribution) > 0.001)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 5);
  const maxAbs = Math.max(...items.map((it) => Math.abs(it.contribution)), 0.0001);

  const list = document.getElementById("factorsList");
  list.innerHTML = "";
  items.forEach((it) => {
    const labelFn = LABELS[it.name] || ((v) => `${it.name}: ${v}`);
    const row = document.createElement("div");
    row.className = "factor-row";
    const widthPct = (Math.abs(it.contribution) / maxAbs) * 48; // half-track max
    const isRisk = it.contribution > 0;
    row.innerHTML = `
      <div class="factor-label">${labelFn(it.value)}</div>
      <div class="factor-track">
        <div class="factor-bar ${isRisk ? "risk" : "protect"}"
             style="width:${widthPct}%; ${isRisk ? `left:50%` : `left:${50 - widthPct}%`}"></div>
      </div>`;
    list.appendChild(row);
  });
}

document.querySelectorAll("input, select").forEach((el) => {
  el.addEventListener("input", render);
  el.addEventListener("change", render);
});
render();
</script>

</body>
</html>
'''

placeholder = '/*__MODEL_PARAMS__*/{"intercept":0,"numeric":[],"categorical":[],"addon_cols":[]}/*__END_MODEL_PARAMS__*/'
final_html = HTML_TEMPLATE.replace(placeholder, json.dumps(model_export))

html_path = os.path.join(SCRIPT_DIR, "churn_app.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(final_html)
print(f"Generated standalone risk-assessment UI: {html_path}")

try:
    import webbrowser
    webbrowser.open(f"file://{html_path}")
except Exception:
    pass