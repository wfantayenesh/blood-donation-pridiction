# =============================================================================
#  File       : app/streamlit_app.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Beautiful Streamlit dashboard — colorful, modern, interactive.
#
#  Run with:
#       streamlit run app/streamlit_app.py
# =============================================================================

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Donation Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Imports ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

  /* ── Background ── */
  .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
  }
  [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
  [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }

  /* ── Hero header ── */
  .hero {
    background: linear-gradient(135deg, #E74C3C 0%, #9B59B6 50%, #3498DB 100%);
    border-radius: 18px;
    padding: 36px 40px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(231,76,60,0.35);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute; top: -40%; right: -10%;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
  }
  .hero h1 {
    font-size: 2.1rem; font-weight: 800;
    color: white; margin: 0 0 6px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }
  .hero p { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 3px 0; }
  .hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white; border-radius: 20px;
    padding: 4px 14px; font-size: 0.82rem;
    margin-right: 8px; margin-top: 10px;
    backdrop-filter: blur(4px);
  }

  /* ── Glass card ── */
  .glass-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 20px;
  }
  .glass-card h3 {
    color: #F1F5F9; font-weight: 700; font-size: 1.1rem;
    margin-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 8px;
  }

  /* ── Metric tile ── */
  .metric-tile {
    background: linear-gradient(135deg, var(--c1), var(--c2));
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
  }
  .metric-tile .val {
    display: block; font-size: 2rem;
    font-weight: 800; color: white;
    text-shadow: 0 2px 6px rgba(0,0,0,0.25);
  }
  .metric-tile .lbl {
    display: block; font-size: 0.78rem;
    color: rgba(255,255,255,0.8);
    margin-top: 4px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em;
  }

  /* ── Section label ── */
  .section-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 10px;
  }

  /* ── Predict button ── */
  .stButton > button {
    background: linear-gradient(135deg, #E74C3C, #9B59B6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    box-shadow: 0 4px 16px rgba(231,76,60,0.4) !important;
    transition: transform 0.15s !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(231,76,60,0.5) !important;
  }

  /* ── Result boxes ── */
  .result-donate {
    background: linear-gradient(135deg, #2ECC71, #1ABC9C);
    border-radius: 14px; padding: 24px 28px;
    text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(46,204,113,0.35);
  }
  .result-no {
    background: linear-gradient(135deg, #E74C3C, #C0392B);
    border-radius: 14px; padding: 24px 28px;
    text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(231,76,60,0.35);
  }
  .result-title { font-size: 1.8rem; font-weight: 800; margin: 0; }
  .result-conf  { font-size: 1rem; opacity: 0.9; margin-top: 6px; }

  /* ── Table ── */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ── Tab styling ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 10px; padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #94A3B8 !important; font-weight: 600;
    border-radius: 8px;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(231,76,60,0.3) !important;
    color: white !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw  = os.path.join(PROJECT_ROOT, "data", "raw", "blood_donation.csv")
    proc = os.path.join(PROJECT_ROOT, "data", "processed", "blood_donation_clean.csv")
    df_raw  = pd.read_csv(raw)  if os.path.exists(raw)  else None
    df_proc = pd.read_csv(proc) if os.path.exists(proc) else None
    if df_raw is not None and "Target" in df_raw.columns:
        df_raw["Target"] = df_raw["Target"].astype(str)
    return df_raw, df_proc


@st.cache_resource
def load_model():
    mp  = os.path.join(PROJECT_ROOT, "models", "model.pkl")
    sp  = os.path.join(PROJECT_ROOT, "models", "preprocessing.pkl")
    mtp = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    model, scaler, meta = None, None, {}
    if os.path.exists(mp):
        with open(mp, "rb") as f: model = pickle.load(f)
    if os.path.exists(sp):
        with open(sp, "rb") as f: scaler = pickle.load(f)
    if os.path.exists(mtp):
        with open(mtp) as f: meta = json.load(f)
    return model, scaler, meta


def make_fig():
    plt.rcParams.update({
        "figure.facecolor": "#1E293B", "axes.facecolor": "#1E293B",
        "text.color": "#E2E8F0", "axes.labelcolor": "#E2E8F0",
        "xtick.color": "#94A3B8", "ytick.color": "#94A3B8",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#334155", "grid.color": "#334155",
        "grid.alpha": 0.4, "grid.linestyle": "--",
    })


def reset_style():
    plt.rcParams.update(plt.rcParamsDefault)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 20px;">
      <div style="font-size:3rem;">🩸</div>
      <div style="font-size:1.1rem; font-weight:800; color:white;">Blood Donation</div>
      <div style="font-size:0.8rem; color:#94A3B8;">Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Dataset Overview", "🔍 EDA & Visualizations",
         "🤖 Model Results", "🔮 Predict Donation"],
        index=0, label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem; color:#94A3B8; line-height:1.9;">
      <b style="color:#CBD5E1;">Authors</b><br>
      Mihret Alemayehu<br>Abebech Nega<br><br>
      <b style="color:#CBD5E1;">Institution</b><br>
      Debre Berhan University<br><br>
      <b style="color:#CBD5E1;">Instructor</b><br>
      Petros Abebe
    </div>
    """, unsafe_allow_html=True)


# ── Data & model ──────────────────────────────────────────────────────────────
df_raw, df_clean = load_data()
model, scaler, metadata = load_model()
TARGET = "Target"

# =============================================================================
# PAGE 1 — Dataset Overview
# =============================================================================
if page == "📊 Dataset Overview":
    st.markdown("""
    <div class="hero">
      <h1>🩸 Blood Donation Prediction System</h1>
      <p>RFMTC Dataset &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; Debre Berhan University</p>
      <span class="badge">Machine Learning</span>
      <span class="badge">Python</span>
      <span class="badge">Scikit-learn</span>
      <span class="badge">Academic Project</span>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Raw dataset not found. Expected at `data/raw/blood_donation.csv`.")
        st.stop()

    donation_rate = (df_raw[TARGET] == "1").sum() / len(df_raw) * 100 if TARGET in df_raw.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, f"{len(df_raw):,}", "Total Records",  "#E74C3C", "#C0392B"),
        (c2, f"{df_raw.shape[1]-1}",    "Features",      "#3498DB", "#2980B9"),
        (c3, f"{df_raw.isnull().sum().sum():,}", "Missing Cells", "#9B59B6", "#8E44AD"),
        (c4, f"{donation_rate:.1f}%",  "Donation Rate", "#2ECC71", "#27AE60"),
    ]
    for col, val, lbl, c1h, c2h in tiles:
        col.markdown(
            f'<div class="metric-tile" style="--c1:{c1h};--c2:{c2h};">'
            f'<span class="val">{val}</span><span class="lbl">{lbl}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    tab_raw, tab_stats, tab_missing = st.tabs(
        ["📋 Raw Data Preview", "📈 Descriptive Statistics", "❗ Missing Values"])

    with tab_raw:
        st.markdown('<p class="section-label">First 20 rows of the raw dataset</p>',
                    unsafe_allow_html=True)
        st.dataframe(df_raw.head(20), use_container_width=True)

    with tab_stats:
        st.markdown('<p class="section-label">Statistical summary (numeric columns)</p>',
                    unsafe_allow_html=True)
        st.dataframe(df_raw.describe().T.round(3), use_container_width=True)

    with tab_missing:
        missing_df = pd.DataFrame({
            "Count": df_raw.isnull().sum(),
            "Percent (%)": (df_raw.isnull().sum() / len(df_raw) * 100).round(2),
        })
        st.dataframe(missing_df, use_container_width=True)


# =============================================================================
# PAGE 2 — EDA & Visualizations
# =============================================================================
elif page == "🔍 EDA & Visualizations":
    st.markdown('<p style="font-size:1.8rem;font-weight:800;color:#F1F5F9;">🔍 Exploratory Data Analysis</p>',
                unsafe_allow_html=True)

    if df_raw is None:
        st.error("Raw dataset not found."); st.stop()

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c != TARGET]

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Distribution", "📦 Box by Target",
         "🌡️ Correlation Heatmap", "🎯 Target Distribution"])

    with tab1:
        col_sel = st.selectbox("Select feature", feat_cols, key="dist_sel")
        make_fig()
        fig, ax = plt.subplots(figsize=(8, 4))
        data = df_raw[col_sel].dropna()
        ax.hist(data, bins=30, color="#3498DB", edgecolor="#1E293B", alpha=0.85, zorder=3)
        ax2 = ax.twinx()
        try:
            data.plot.kde(ax=ax2, color="#E74C3C", linewidth=2.5)
        except Exception:
            pass
        ax2.set_yticks([]); ax2.set_ylabel("")
        ax2.spines["right"].set_visible(False)
        ax.set_title(f"Distribution of  {col_sel}", color="#F1F5F9", fontsize=13)
        ax.set_xlabel(col_sel); ax.set_ylabel("Count")
        st.pyplot(fig); plt.close(fig)
        reset_style()

    with tab2:
        col_sel2 = st.selectbox("Select feature", feat_cols, key="box_sel")
        make_fig()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=TARGET, y=col_sel2, data=df_raw,
                    palette={"0": "#E74C3C", "1": "#3498DB"}, ax=ax,
                    linewidth=1.5, fliersize=4)
        ax.set_xticklabels(["Won't Donate (0)", "Will Donate (1)"])
        ax.set_title(f"{col_sel2}  by  Target Class",
                     color="#F1F5F9", fontsize=13)
        st.pyplot(fig); plt.close(fig)
        reset_style()

    with tab3:
        make_fig()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")
        cmap = sns.diverging_palette(230, 15, s=85, l=40, as_cmap=True)
        num_df = df_raw.select_dtypes(include=[np.number])
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap=cmap,
                    center=0, linewidths=1, linecolor="#1E293B",
                    square=True, ax=ax,
                    annot_kws={"size": 9, "weight": "bold", "color": "white"})
        ax.set_title("Pearson Correlation Matrix", fontsize=14,
                     color="white", pad=16)
        ax.tick_params(colors="white")
        st.pyplot(fig); plt.close(fig)
        reset_style()

    with tab4:
        make_fig()
        counts = df_raw[TARGET].value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.patch.set_facecolor("#1E293B")
        ax1.set_facecolor("#1E293B"); ax2.set_facecolor("#1E293B")
        bars = ax1.bar(["Won't Donate", "Will Donate"],
                       counts.values, color=["#E74C3C", "#3498DB"],
                       edgecolor="#1E293B", width=0.5)
        total = counts.sum()
        for bar, cnt in zip(bars, counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + total*0.02,
                     f"{cnt:,}\n({cnt/total*100:.1f}%)",
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold", color="white")
        ax1.set_title("Class Count", color="#F1F5F9", fontsize=13)
        ax1.set_ylabel("Count", color="#94A3B8")
        ax2.pie(counts.values, labels=["Won't Donate", "Will Donate"],
                colors=["#E74C3C", "#3498DB"],
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"width": 0.55, "edgecolor": "#1E293B", "linewidth": 3},
                pctdistance=0.75,
                textprops={"fontsize": 11, "color": "white"})
        ax2.set_title("Class Proportion", color="#F1F5F9", fontsize=13)
        st.pyplot(fig); plt.close(fig)
        reset_style()


# =============================================================================
# PAGE 3 — Model Results
# =============================================================================
elif page == "🤖 Model Results":
    st.markdown('<p style="font-size:1.8rem;font-weight:800;color:#F1F5F9;">🤖 Model Evaluation</p>',
                unsafe_allow_html=True)

    if model is None or df_clean is None:
        st.warning("Model or processed dataset not found. Run the pipeline first.")
        st.stop()

    y      = df_clean[TARGET]
    X      = df_clean.drop(columns=[TARGET])
    Xs     = pd.DataFrame(scaler.transform(X), columns=X.columns)
    _, Xt, _, yt = train_test_split(Xs, y, test_size=0.20, random_state=42, stratify=y)

    yp   = model.predict(Xt)
    yprb = model.predict_proba(Xt)[:, 1]

    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)
    auc  = roc_auc_score(yt, yprb)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl, ca, cb in [
        (c1, f"{acc:.2%}",  "Accuracy",  "#3498DB", "#2980B9"),
        (c2, f"{auc:.4f}",  "ROC-AUC",   "#9B59B6", "#8E44AD"),
        (c3, f"{f1:.4f}",   "F1 Score",  "#E74C3C", "#C0392B"),
        (c4, f"{prec:.4f}", "Precision", "#F39C12", "#D68910"),
        (c5, f"{rec:.4f}",  "Recall",    "#2ECC71", "#27AE60"),
    ]:
        col.markdown(
            f'<div class="metric-tile" style="--c1:{ca};--c2:{cb};">'
            f'<span class="val">{val}</span><span class="lbl">{lbl}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(yt, yp)
        make_fig()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        fig.patch.set_facecolor("#0F172A"); ax.set_facecolor("#0F172A")
        cmap_c = LinearSegmentedColormap.from_list(
            "db", ["#1E293B", "#1D4ED8", "#60A5FA"], N=256)
        ax.imshow(cm, cmap=cmap_c, aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Won't\nDonate","Will\nDonate"], color="white", fontsize=11)
        ax.set_yticklabels(["Won't\nDonate","Will\nDonate"], color="white", fontsize=11)
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        ax.tick_params(colors="white")
        lbl_map = [["TN","FP"],["FN","TP"]]
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{lbl_map[r][c]}\n{cm[r,c]:,}",
                        ha="center", va="center", fontsize=16,
                        fontweight="bold", color="white")
        ax.set_title("Confusion Matrix", color="white", fontsize=13)
        for sp in ax.spines.values(): sp.set_visible(False)
        st.pyplot(fig); plt.close(fig)
        reset_style()

    with col_roc:
        st.markdown('<p class="section-label">ROC Curve</p>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(yt, yprb)
        make_fig()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, color="#3498DB", lw=2.5,
                label=f"AUC = {auc:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.15, color="#3498DB")
        ax.plot([0,1],[0,1],"--", color="#64748B", lw=1.2, label="Random (0.500)")
        ax.text(0.55, 0.1, f"AUC={auc:.3f}", fontsize=18,
                fontweight="bold", color="#3498DB", alpha=0.35)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve", fontsize=13)
        ax.legend(fontsize=10)
        st.pyplot(fig); plt.close(fig)
        reset_style()

    st.markdown('<p class="section-label" style="margin-top:20px;">Feature Importances</p>',
                unsafe_allow_html=True)
    fi_df = pd.DataFrame({
        "feature": X.columns, "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    make_fig()
    cmap = plt.cm.RdYlGn
    colors = [cmap(v / fi_df["importance"].max()) for v in fi_df["importance"]]
    fig, ax = plt.subplots(figsize=(10, max(4, len(fi_df)*0.38)))
    ax.barh(fi_df["feature"], fi_df["importance"],
            color=colors, edgecolor="#1E293B", height=0.65)
    for bar, val in zip(ax.patches, fi_df["importance"]):
        ax.text(bar.get_width() + fi_df["importance"].max()*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left", fontsize=9,
                fontweight="bold", color="#E2E8F0")
    ax.set_xlabel("Gini Importance")
    ax.set_title("Feature Importances — Random Forest", fontsize=13)
    ax.set_xlim(0, fi_df["importance"].max() * 1.18)
    st.pyplot(fig); plt.close(fig)
    reset_style()

    with st.expander("📋 Full Classification Report"):
        report = classification_report(yt, yp,
                                       target_names=["Won't Donate", "Will Donate"])
        st.code(report)


# =============================================================================
# PAGE 4 — Predict
# =============================================================================
elif page == "🔮 Predict Donation":
    st.markdown('<p style="font-size:1.8rem;font-weight:800;color:#F1F5F9;">🔮 Predict Donation</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#94A3B8;">Enter donor RFMTC values to get a prediction.</p>',
                unsafe_allow_html=True)

    if model is None or scaler is None or df_clean is None:
        st.warning("Model artifacts not found. Run the pipeline first."); st.stop()

    feature_cols = [c for c in df_clean.columns if c != TARGET]

    st.markdown('<p class="section-label">Donor Feature Inputs</p>', unsafe_allow_html=True)

    user_input = {}
    cols = st.columns(3)
    for i, feature in enumerate(feature_cols):
        vals = df_clean[feature]
        user_input[feature] = cols[i % 3].number_input(
            label=feature,
            min_value=float(vals.min()),
            max_value=float(vals.max()),
            value=round(float(vals.mean()), 2),
            step=0.01,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict", type="primary"):
        inp_df     = pd.DataFrame([user_input])
        inp_scaled = scaler.transform(inp_df)
        pred       = model.predict(inp_scaled)[0]
        probs      = model.predict_proba(inp_scaled)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown(
                f'<div class="result-donate">'
                f'<p class="result-title">✅ Will Donate</p>'
                f'<p class="result-conf">Confidence: {probs[1]:.1%}</p>'
                f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="result-no">'
                f'<p class="result-title">❌ Won\'t Donate</p>'
                f'<p class="result-conf">Confidence: {probs[0]:.1%}</p>'
                f'</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        make_fig()
        fig, ax = plt.subplots(figsize=(5, 3))
        bar_colors = ["#E74C3C", "#2ECC71"]
        bars = ax.bar(["Won't Donate", "Will Donate"], probs,
                      color=bar_colors, edgecolor="#1E293B", width=0.45)
        for bar, val in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02, f"{val:.1%}",
                    ha="center", va="bottom", fontsize=13,
                    fontweight="bold", color="#E2E8F0")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probabilities", fontsize=13)
        st.pyplot(fig); plt.close(fig)
        reset_style()
