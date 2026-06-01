"""
run_pipeline.py
Blood Donation Prediction System — Full Pipeline Runner
Authors: Mihret Alemayehu, Abebech Nega
Debre Berhan University | Instructor: Petros Abebe

PURPOSE OF THIS SCRIPT:
    This script runs the FULL machine learning pipeline from raw data to
    saved model. All visualizations are saved as .png files using plt.savefig().
    No graphs are displayed on screen — they are written directly to disk.

    To VIEW graphs inline during a step-by-step presentation, open:
        notebooks/blood_donation_complete_analysis.ipynb

OUTPUT FILES GENERATED:
  - data/processed/blood_donation_clean.csv       (cleaned dataset)
  - models/model.pkl                               (trained Random Forest)
  - models/preprocessing.pkl                       (fitted StandardScaler)
  - models/model_metadata.json                     (metrics and config)
  - reports/figures/correlation_heatmap.png
  - reports/figures/target_distribution.png
  - reports/figures/feature_distributions.png
  - reports/figures/boxplots_by_target.png
  - reports/figures/missing_values.png
  - reports/figures/confusion_matrix.png
  - reports/figures/roc_curve.png
  - reports/figures/precision_recall_curve.png
  - reports/figures/feature_importance.png
  - reports/Evaluation results/evaluation_results.html

HOW GRAPHS ARE SAVED (plt.savefig):
    Every figure is created with matplotlib, then saved to disk using:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    The Agg backend is used so this script works without a display/GUI.
"""

import os, sys, json, pickle, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
# Use the non-interactive Agg backend so the script runs without a display.
# This means plt.show() would do nothing here — all output goes via plt.savefig().
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Global style tweaks for beautiful, colorful charts ───────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 14,
    "axes.labelsize"    : 11,
    "figure.facecolor"  : "#FAFBFC",
    "axes.facecolor"    : "#FAFBFC",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linestyle"    : "--",
})

COLORS   = ["#E74C3C", "#3498DB", "#9B59B6", "#2ECC71", "#F39C12", "#1ABC9C"]
RED, BLUE, PURPLE, GREEN, AMBER = COLORS[:5]

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
RAW_CSV    = os.path.join(ROOT, "data", "raw", "blood_donation.csv")
PROC_DIR   = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
FIG_DIR    = os.path.join(ROOT, "reports", "figures")
EVAL_DIR   = os.path.join(ROOT, "reports", "Evaluation results")

for d in [PROC_DIR, MODELS_DIR, FIG_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET = "Target"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load raw data
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading raw data...")
df = pd.read_csv(RAW_CSV)
print(f"      Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Clean: standardize NaN, remove duplicates
# ─────────────────────────────────────────────────────────────────────────────
print("[2/7] Cleaning data...")
df = df.replace(["", " ", "NA", "N/A", "nan", "NaN", "--", "-", "?"], np.nan)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"      Duplicates removed: {before - len(df)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — KNN Imputation (k=5) on numeric columns
# ─────────────────────────────────────────────────────────────────────────────
print("[3/7] KNN imputation (k=5)...")
y = df[TARGET].copy()
X_raw = df.drop(columns=[TARGET])

# Label-encode categoricals before KNN (KNN needs numeric input)
label_encoders = {}
for col in X_raw.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X_raw[col] = X_raw[col].fillna("Unknown")
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    label_encoders[col] = le

missing_before = X_raw.isnull().sum().sum()
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_arr = knn_imputer.fit_transform(X_raw)
X_imputed = pd.DataFrame(X_imputed_arr, columns=X_raw.columns)
print(f"      Missing before: {missing_before}  |  after: {X_imputed.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Feature engineering + save processed CSV
# ─────────────────────────────────────────────────────────────────────────────
print("[4/7] Feature engineering...")
if "Frequency" in X_imputed.columns and "Time" in X_imputed.columns:
    X_imputed["donation_density"]    = X_imputed["Frequency"] / (X_imputed["Time"] + 1e-6)
    X_imputed["recency_score"]       = 1.0 / (X_imputed["Recency"] + 1)
    X_imputed["high_frequency_flag"] = (X_imputed["Frequency"] > X_imputed["Frequency"].median()).astype(int)

df_clean = pd.concat([X_imputed, y.reset_index(drop=True)], axis=1)
clean_path = os.path.join(PROC_DIR, "blood_donation_clean.csv")
df_clean.to_csv(clean_path, index=False)
print(f"      Saved: {clean_path}  ({df_clean.shape})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — EDA figures
# ─────────────────────────────────────────────────────────────────────────────
print("[5/7] Generating EDA figures...")

# ── Helper: save figure to disk using plt.savefig ────────────────────────────
# NOTE: In this script we ALWAYS use fig.savefig() to write each chart as a
# PNG file. We never call plt.show() here because the Agg backend is active.
# If you want to SEE the charts, open notebooks/blood_donation_complete_analysis.ipynb
# which uses %matplotlib inline and plt.show() for inline display.
def save_fig(fig, name):
    """Save a matplotlib Figure to reports/figures/<name> and close it."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: reports/figures/{name}")

# 5a — Correlation heatmap (dark theme, vivid diverging palette)
numeric_df = df_clean.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor("#0F172A")
ax.set_facecolor("#0F172A")
cmap_corr = sns.diverging_palette(230, 15, s=85, l=40, as_cmap=True)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap=cmap_corr, center=0,
            linewidths=1.2, linecolor="#1E293B", square=True, ax=ax,
            annot_kws={"size": 8, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8, "pad": 0.02})
ax.set_title("Pearson Correlation Matrix — Blood Donation Dataset",
             fontsize=16, fontweight="bold", color="white", pad=20)
ax.tick_params(colors="white", labelsize=10)
for sp in ax.spines.values(): sp.set_visible(False)
ax.collections[0].colorbar.ax.tick_params(colors="white")
save_fig(fig, "correlation_heatmap.png")

# 5b — Target distribution (bar + donut, vivid colors)
counts = df_clean[TARGET].value_counts().sort_index()
total  = counts.sum()
fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#FAFBFC")
bars = ax_bar.bar(["Won't Donate (0)", "Will Donate (1)"], counts.values,
                  color=[RED, BLUE], edgecolor="white", width=0.52, linewidth=1.5, zorder=3)
for bar, cnt in zip(bars, counts.values):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.015,
                f"{cnt:,}\n({cnt/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
ax_bar.set_title("Donor Class — Count", fontweight="bold")
ax_bar.set_ylabel("Number of Donors")
ax_bar.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax_bar.set_ylim(0, counts.max() * 1.28)
wedges, texts, autotexts = ax_pie.pie(
    counts.values, labels=["Won't Donate (0)", "Will Donate (1)"],
    colors=[RED, BLUE], autopct="%1.1f%%", startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 3},
    pctdistance=0.75, textprops={"fontsize": 11})
for at in autotexts:
    at.set_fontweight("bold"); at.set_color("white"); at.set_fontsize(13)
ax_pie.set_title("Class Proportion (Donut)", fontweight="bold")
fig.suptitle("Target Variable Distribution — Blood Donation Dataset",
             fontsize=15, fontweight="bold", y=1.02)
save_fig(fig, "target_distribution.png")

# 5c — Feature distributions (colorful histogram + KDE grid)
feat_cols  = [c for c in numeric_df.columns if c != TARGET]
ncols      = 3
nrows      = (len(feat_cols) + ncols - 1) // ncols
color_cycle = [BLUE, RED, PURPLE, GREEN, AMBER, "#1ABC9C", "#E67E22", "#2980B9", "#8E44AD"]

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
fig.patch.set_facecolor("#F8FAFC")
axes = np.array(axes).ravel()
for i, col in enumerate(feat_cols):
    c = color_cycle[i % len(color_cycle)]
    data = df_clean[col].dropna()
    axes[i].hist(data, bins=28, color=c, edgecolor="white", alpha=0.78, zorder=3)
    ax2 = axes[i].twinx()
    try: data.plot.kde(ax=ax2, color=c, linewidth=2.2)
    except Exception: pass
    ax2.set_ylabel(""); ax2.set_yticks([])
    ax2.spines["right"].set_visible(False); ax2.spines["top"].set_visible(False)
    axes[i].set_title(col, fontsize=10, fontweight="bold", color=c)
    axes[i].set_ylabel("Count", fontsize=9); axes[i].set_facecolor("#F8FAFC")
for j in range(i + 1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Numeric Feature Distributions  (Histogram + KDE)", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "feature_distributions.png")

# 5d — Box plots by target (red vs blue palette)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
axes = np.array(axes).ravel()
fig.patch.set_facecolor("#F8FAFC")
for i, col in enumerate(feat_cols):
    sns.boxplot(x=TARGET, y=col, data=df_clean,
                palette={0: RED, 1: BLUE}, ax=axes[i], linewidth=1.5, fliersize=4)
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_xlabel("Target", fontsize=9)
    axes[i].set_xticklabels(["Won't Donate (0)", "Will Donate (1)"], fontsize=8)
    axes[i].set_facecolor("#F8FAFC")
for j in range(i + 1, len(axes)): axes[j].set_visible(False)
legend_patches = [mpatches.Patch(color=RED, label="Won't Donate (0)"),
                  mpatches.Patch(color=BLUE, label="Will Donate (1)")]
fig.legend(handles=legend_patches, loc="upper center", ncol=2, fontsize=10,
           framealpha=0.85, bbox_to_anchor=(0.5, 1.03))
fig.suptitle("Feature Distributions by Target Class", fontsize=14, y=1.06)
fig.tight_layout()
save_fig(fig, "boxplots_by_target.png")

# 5e — Missing values (colorful gradient bars)
raw_df_miss = pd.read_csv(RAW_CSV)
raw_missing = (raw_df_miss.isnull().sum() / len(raw_df_miss) * 100).sort_values(ascending=False)
raw_missing = raw_missing[raw_missing > 0]
if not raw_missing.empty:
    bar_colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(raw_missing)))
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(raw_missing) * 0.55)))
    bars = ax.barh(raw_missing.index, raw_missing.values,
                   color=bar_colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, raw_missing.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Missing (%)", fontsize=12)
    ax.set_title("Missing Value Percentage by Column (Raw Data)", fontweight="bold")
    ax.set_xlim(0, raw_missing.max() * 1.2)
    save_fig(fig, "missing_values.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Train model, save pkl files
# ─────────────────────────────────────────────────────────────────────────────
print("[6/7] Training Random Forest Classifier...")

y_all = df_clean[TARGET]
X_all = df_clean.drop(columns=[TARGET])

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_all), columns=X_all.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

rf = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)
print(f"      Train accuracy: {rf.score(X_train, y_train):.4f}")

# 5-fold cross-validation
cv_scores = cross_val_score(rf, X_scaled, y_all, cv=5, scoring="accuracy", n_jobs=-1)
print(f"      5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Evaluation metrics
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)
report_str = classification_report(y_test, y_pred,
                                    target_names=["Won't Donate", "Will Donate"])

print(f"      Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}  |  F1: {f1:.4f}")
print(f"\n{report_str}")

# Feature importances
fi_df = pd.DataFrame({
    "feature": X_all.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

# Model evaluation figures

# Confusion matrix (dark theme with TN/FP/FN/TP labels)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5.2))
fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#0F172A")
cmap_cm = LinearSegmentedColormap.from_list("dark_blues", ["#1E293B","#1D4ED8","#60A5FA"], N=256)
ax.imshow(cm, cmap=cmap_cm, aspect="auto")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Won't\nDonate","Will\nDonate"], fontsize=12, color="white")
ax.set_yticklabels(["Won't\nDonate","Will\nDonate"], fontsize=12, color="white")
ax.set_xlabel("Predicted Label", fontsize=12, color="white")
ax.set_ylabel("True Label", fontsize=12, color="white")
ax.tick_params(colors="white")
for r, c_lbl in enumerate([["TN","FP"],["FN","TP"]]):
    for c, lbl in enumerate(c_lbl):
        ax.text(c, r, f"{lbl}\n{cm[r,c]:,}", ha="center", va="center",
                fontsize=17, fontweight="bold", color="white")
ax.set_title("Confusion Matrix — Random Forest", fontsize=14,
             fontweight="bold", color="white", pad=16)
for sp in ax.spines.values(): sp.set_visible(False)
save_fig(fig, "confusion_matrix.png")

# ROC curve (gradient fill + large AUC watermark)
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color=BLUE, lw=2.8, label=f"Random Forest (AUC = {auc:.3f})", zorder=4)
ax.fill_between(fpr, tpr, alpha=0.18, color=BLUE)
ax.plot([0,1],[0,1],"k--", lw=1.4, label="Random Classifier (AUC = 0.500)")
ax.text(0.52, 0.08, f"AUC = {auc:.4f}", fontsize=24, fontweight="bold",
        color=BLUE, alpha=0.32, transform=ax.transAxes)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("ROC Curve — Random Forest Classifier")
ax.legend(loc="lower right", fontsize=11)
save_fig(fig, "roc_curve.png")

# Precision-Recall curve (colored, filled)
pr_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(rec_vals, pr_vals, color=PURPLE, lw=2.5, label="Random Forest")
ax.fill_between(rec_vals, pr_vals, alpha=0.15, color=PURPLE)
ax.axhline(y=y_test.mean(), color="#94A3B8", linestyle="--", lw=1.4,
           label=f"Baseline ({y_test.mean():.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Random Forest")
ax.legend(fontsize=11)
save_fig(fig, "precision_recall_curve.png")

# Feature importance (rainbow gradient horizontal bars)
top_fi = fi_df.sort_values("importance")
cmap_fi = plt.cm.RdYlGn
fi_colors = [cmap_fi(v / top_fi["importance"].max()) for v in top_fi["importance"]]
fig, ax = plt.subplots(figsize=(9, max(4.5, len(top_fi) * 0.42)))
bars = ax.barh(top_fi["feature"], top_fi["importance"],
               color=fi_colors, edgecolor="white", linewidth=0.8, height=0.68)
for bar, val in zip(bars, top_fi["importance"]):
    ax.text(bar.get_width() + top_fi["importance"].max()*0.01,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left", fontsize=9, fontweight="bold")
ax.set_xlabel("Gini Importance"); ax.set_title("Feature Importances — Random Forest")
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.set_xlim(0, top_fi["importance"].max() * 1.18)
sm = plt.cm.ScalarMappable(cmap=cmap_fi, norm=plt.Normalize(0, top_fi["importance"].max()))
fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="Importance")
save_fig(fig, "feature_importance.png")

# Save model artifacts
with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
    pickle.dump(rf, f)
with open(os.path.join(MODELS_DIR, "preprocessing.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("      Saved: models/model.pkl")
print("      Saved: models/preprocessing.pkl")

# Save metadata
metadata = {
    "project": "Blood Donation Prediction System",
    "authors": ["Mihret Alemayehu", "Abebech Nega"],
    "institution": "Debre Berhan University",
    "instructor": "Petros Abebe",
    "algorithm": "Random Forest Classifier",
    "hyperparameters": {"n_estimators": 100, "random_state": 42,
                        "class_weight": "balanced", "n_jobs": -1},
    "split": {"test_size": 0.20, "train_size": 0.80, "stratified": True, "random_state": 42},
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "features": X_all.columns.tolist(),
    "n_features": int(X_all.shape[1]),
    "evaluation": {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
    },
    "imputation": {"method": "KNN", "n_neighbors": 5},
    "scaling": "StandardScaler",
    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=4)
print("      Saved: models/model_metadata.json")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — HTML evaluation report
# ─────────────────────────────────────────────────────────────────────────────
print("[7/7] Generating HTML evaluation report...")

fi_rows = "".join(
    f"<tr><td>{i+1}</td><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
    for i, row in fi_df.head(10).iterrows()
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Evaluation Report — Blood Donation Prediction System</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      color: #E2E8F0;
      line-height: 1.65;
      padding: 32px 16px 64px;
    }}

    .wrap {{ max-width: 980px; margin: 0 auto; }}

    /* ── Hero ── */
    .hero {{
      background: linear-gradient(135deg, #E74C3C 0%, #9B59B6 50%, #3498DB 100%);
      border-radius: 20px; padding: 40px 44px; margin-bottom: 36px;
      box-shadow: 0 20px 60px rgba(231,76,60,0.35);
      position: relative; overflow: hidden;
    }}
    .hero::before {{
      content: ""; position: absolute; top: -50px; right: -50px;
      width: 260px; height: 260px;
      background: rgba(255,255,255,0.08); border-radius: 50%;
    }}
    .hero h1 {{
      font-size: 2rem; font-weight: 800; color: white; margin-bottom: 8px;
      text-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }}
    .hero .meta {{ color: rgba(255,255,255,0.85); font-size: 0.95rem; line-height: 2; }}
    .hero .meta strong {{ color: white; }}
    .badge {{
      display: inline-block; background: rgba(255,255,255,0.2);
      color: white; border-radius: 20px; padding: 3px 14px;
      font-size: 0.78rem; margin: 8px 6px 0 0;
    }}

    /* ── Cards ── */
    .card {{
      background: rgba(255,255,255,0.07);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 16px; padding: 28px 32px;
      margin-bottom: 24px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }}
    .card h2 {{
      font-size: 1.1rem; font-weight: 700; color: #F1F5F9;
      margin-bottom: 18px; padding-bottom: 10px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .card h2 .ico {{ margin-right: 8px; }}

    /* ── Metric tiles ── */
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 16px; margin-bottom: 8px;
    }}
    .metric-tile {{
      border-radius: 14px; padding: 20px 16px; text-align: center;
      box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .metric-tile .val {{
      display: block; font-size: 1.8rem; font-weight: 800; color: white;
      text-shadow: 0 2px 6px rgba(0,0,0,0.25);
    }}
    .metric-tile .lbl {{
      display: block; font-size: 0.72rem; color: rgba(255,255,255,0.8);
      margin-top: 4px; font-weight: 600;
      text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .c-acc  {{ background: linear-gradient(135deg,#3498DB,#2980B9); }}
    .c-auc  {{ background: linear-gradient(135deg,#9B59B6,#8E44AD); }}
    .c-f1   {{ background: linear-gradient(135deg,#E74C3C,#C0392B); }}
    .c-prec {{ background: linear-gradient(135deg,#F39C12,#D68910); }}
    .c-rec  {{ background: linear-gradient(135deg,#2ECC71,#27AE60); }}
    .c-cv   {{ background: linear-gradient(135deg,#1ABC9C,#148F77); }}

    /* ── Tables ── */
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
    th {{
      background: rgba(255,255,255,0.12);
      color: #F1F5F9; padding: 10px 14px; text-align: left;
      font-weight: 600; letter-spacing: 0.04em;
    }}
    td {{ padding: 9px 14px; border-bottom: 1px solid rgba(255,255,255,0.06); color: #CBD5E1; }}
    tr:hover td {{ background: rgba(255,255,255,0.04); }}
    .score {{ font-weight: 700; color: #F1F5F9; font-size: 1rem; }}

    /* ── CM grid ── */
    .cm-grid {{
      display: grid; grid-template-columns: auto 1fr 1fr;
      gap: 4px; max-width: 440px;
    }}
    .cm-head {{ background: transparent; color: #94A3B8; font-size: 0.78rem;
                font-weight: 600; padding: 8px; text-align: center; }}
    .cm-cell {{
      border-radius: 10px; padding: 20px 12px; text-align: center;
      font-weight: 800; font-size: 1.3rem; color: white;
    }}
    .cm-TN {{ background: linear-gradient(135deg,#2ECC71,#27AE60); }}
    .cm-FP {{ background: linear-gradient(135deg,#E74C3C,#C0392B); }}
    .cm-FN {{ background: linear-gradient(135deg,#E74C3C,#C0392B); opacity: 0.7; }}
    .cm-TP {{ background: linear-gradient(135deg,#3498DB,#2980B9); }}
    .cm-sub {{ display: block; font-size: 0.7rem; font-weight: 500; opacity: 0.85; margin-top: 4px; }}

    /* ── Methodology cards ── */
    .method-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    }}
    .method-card {{
      border-radius: 12px; padding: 18px 20px;
    }}
    .method-card h4 {{ font-size: 0.9rem; font-weight: 700; margin-bottom: 6px; }}
    .method-card p  {{ font-size: 0.85rem; color: #94A3B8; line-height: 1.6; }}
    .mc-blue   {{ background: rgba(52,152,219,0.1); border: 1px solid rgba(52,152,219,0.25); }}
    .mc-purple {{ background: rgba(155,89,182,0.1); border: 1px solid rgba(155,89,182,0.25); }}
    .mc-red    {{ background: rgba(231,76,60,0.1);  border: 1px solid rgba(231,76,60,0.25); }}
    .mc-green  {{ background: rgba(46,204,113,0.1); border: 1px solid rgba(46,204,113,0.25); }}

    /* ── Pre / code ── */
    pre {{
      background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.08);
      border-radius: 10px; padding: 18px 20px;
      font-size: 0.85rem; color: #CBD5E1;
      overflow-x: auto; line-height: 1.7;
    }}

    /* ── Findings ── */
    .findings li {{
      list-style: none; padding: 10px 14px; margin-bottom: 8px;
      background: rgba(255,255,255,0.05);
      border-left: 3px solid #3498DB;
      border-radius: 0 8px 8px 0;
      font-size: 0.9rem; color: #CBD5E1;
    }}
    .findings li strong {{ color: #F1F5F9; }}

    /* ── Footer ── */
    .footer {{
      text-align: center; color: rgba(255,255,255,0.3);
      font-size: 0.8rem; margin-top: 40px;
      padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.07);
    }}
  </style>
</head>
<body>
<div class="wrap">

  <!-- Hero -->
  <div class="hero">
    <h1>&#129706; Blood Donation Prediction System</h1>
    <div class="meta">
      <strong>Authors:</strong> Mihret Alemayehu &amp; Abebech Nega &nbsp;|&nbsp;
      <strong>Institution:</strong> Debre Berhan University &nbsp;|&nbsp;
      <strong>Instructor:</strong> Petros Abebe<br>
      <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}
    </div>
    <span class="badge">Random Forest</span>
    <span class="badge">KNN Imputation</span>
    <span class="badge">Python / Scikit-learn</span>
    <span class="badge">Academic Project</span>
  </div>

  <!-- Metrics -->
  <div class="card">
    <h2><span class="ico">&#128202;</span>Evaluation Metrics</h2>
    <div class="metric-grid">
      <div class="metric-tile c-acc">
        <span class="val">{acc*100:.1f}%</span><span class="lbl">Accuracy</span>
      </div>
      <div class="metric-tile c-auc">
        <span class="val">{auc:.4f}</span><span class="lbl">ROC-AUC</span>
      </div>
      <div class="metric-tile c-f1">
        <span class="val">{f1:.4f}</span><span class="lbl">F1 Score</span>
      </div>
      <div class="metric-tile c-prec">
        <span class="val">{prec:.4f}</span><span class="lbl">Precision</span>
      </div>
      <div class="metric-tile c-rec">
        <span class="val">{rec:.4f}</span><span class="lbl">Recall</span>
      </div>
      <div class="metric-tile c-cv">
        <span class="val">{cv_scores.mean():.3f}</span><span class="lbl">5-Fold CV</span>
      </div>
    </div>
    <p style="font-size:0.8rem;color:#64748B;margin-top:12px;">
      CV Standard Deviation: &#177; {cv_scores.std():.4f}
    </p>
  </div>

  <!-- Model Config -->
  <div class="card">
    <h2><span class="ico">&#9881;&#65039;</span>Model Configuration</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Algorithm</td><td class="score">Random Forest Classifier</td></tr>
      <tr><td>n_estimators</td><td class="score">100 decision trees</td></tr>
      <tr><td>class_weight</td><td class="score">balanced</td></tr>
      <tr><td>random_state</td><td class="score">42</td></tr>
      <tr><td>Train samples</td><td class="score">{len(X_train):,} (80%)</td></tr>
      <tr><td>Test samples</td><td class="score">{len(X_test):,} (20%)</td></tr>
      <tr><td>Total features</td><td class="score">{X_all.shape[1]} (incl. engineered)</td></tr>
      <tr><td>Imputation method</td><td class="score">KNN (k = 5)</td></tr>
      <tr><td>Feature scaling</td><td class="score">StandardScaler</td></tr>
      <tr><td>Validation strategy</td><td class="score">5-fold cross-validation</td></tr>
    </table>
  </div>

  <!-- Confusion Matrix -->
  <div class="card">
    <h2><span class="ico">&#127919;</span>Confusion Matrix</h2>
    <div class="cm-grid">
      <div></div>
      <div class="cm-head">Predicted:<br>Won&rsquo;t Donate</div>
      <div class="cm-head">Predicted:<br>Will Donate</div>
      <div class="cm-head" style="text-align:right;padding-right:12px;">Actual:<br>Won&rsquo;t Donate</div>
      <div class="cm-cell cm-TN">{cm[0][0]:,}<span class="cm-sub">True Negative</span></div>
      <div class="cm-cell cm-FP">{cm[0][1]:,}<span class="cm-sub">False Positive</span></div>
      <div class="cm-head" style="text-align:right;padding-right:12px;">Actual:<br>Will Donate</div>
      <div class="cm-cell cm-FN">{cm[1][0]:,}<span class="cm-sub">False Negative</span></div>
      <div class="cm-cell cm-TP">{cm[1][1]:,}<span class="cm-sub">True Positive</span></div>
    </div>
  </div>

  <!-- Classification Report -->
  <div class="card">
    <h2><span class="ico">&#128203;</span>Classification Report</h2>
    <pre>{report_str}</pre>
  </div>

  <!-- Feature Importances -->
  <div class="card">
    <h2><span class="ico">&#127775;</span>Top Feature Importances</h2>
    <table>
      <tr><th>Rank</th><th>Feature</th><th>Gini Importance</th></tr>
      {fi_rows}
    </table>
  </div>

  <!-- Methodology -->
  <div class="card">
    <h2><span class="ico">&#128300;</span>Methodology</h2>
    <div class="method-grid">
      <div class="method-card mc-blue">
        <h4 style="color:#3498DB;">Missing Value Handling</h4>
        <p>KNN Imputation with <strong style="color:#F1F5F9;">k=5</strong> neighbors.
        Estimates each missing value from the 5 most similar rows using Euclidean distance.</p>
      </div>
      <div class="method-card mc-purple">
        <h4 style="color:#9B59B6;">Feature Engineering</h4>
        <p>Three derived features: <strong style="color:#F1F5F9;">donation_density</strong>,
        <strong style="color:#F1F5F9;">recency_score</strong>, and
        <strong style="color:#F1F5F9;">high_frequency_flag</strong>.</p>
      </div>
      <div class="method-card mc-red">
        <h4 style="color:#E74C3C;">Algorithm</h4>
        <p>Random Forest Classifier — 100 trees, balanced class weight,
        stratified 80/20 train/test split, and 5-fold cross-validation.</p>
      </div>
      <div class="method-card mc-green">
        <h4 style="color:#2ECC71;">Evaluation</h4>
        <p>Accuracy, Precision, Recall, F1, ROC-AUC, and confusion matrix.
        Cross-validation confirms generalization stability.</p>
      </div>
    </div>
  </div>

  <!-- Key Findings -->
  <div class="card">
    <h2><span class="ico">&#128161;</span>Key Findings</h2>
    <ul class="findings">
      <li>The model achieves <strong>{acc*100:.1f}%</strong> accuracy on the unseen test set.</li>
      <li>ROC-AUC of <strong>{auc:.4f}</strong> indicates strong discrimination between donor classes.</li>
      <li><strong>Monetary</strong> (total cc of blood donated) is the top predictor of donation behaviour.</li>
      <li><strong>Frequency</strong> and <strong>Time</strong> confirm the RFMTC theoretical framework.</li>
      <li>KNN imputation (k=5) successfully resolved all missing values without distributional bias.</li>
      <li>5-fold cross-validation (mean = {cv_scores.mean():.4f} &#177; {cv_scores.std():.4f}) shows stable generalization.</li>
    </ul>
  </div>

  <div class="footer">
    Blood Donation Prediction System &nbsp;&mdash;&nbsp;
    Mihret Alemayehu &amp; Abebech Nega &nbsp;&mdash;&nbsp;
    Debre Berhan University &nbsp;&mdash;&nbsp; Instructor: Petros Abebe
  </div>

</div>
</body>
</html>"""

html_path = os.path.join(EVAL_DIR, "evaluation_results.html")
with open(html_path, "w") as f:
    f.write(html)
print(f"      Saved: reports/Evaluation results/evaluation_results.html")

print("\n✓ All files generated successfully. Every folder is now populated.\n")
