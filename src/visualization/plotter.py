# =============================================================================
#  File       : src/visualization/plotter.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Beautiful, colorful plotting utilities for EDA, model
#               evaluation visuals, and feature importance charts.
#               All figures are saved to reports/figures/.
# =============================================================================

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE_RED    = "#E74C3C"
PALETTE_BLUE   = "#3498DB"
PALETTE_PURPLE = "#9B59B6"
PALETTE_GREEN  = "#2ECC71"
PALETTE_AMBER  = "#F39C12"
PALETTE_TEAL   = "#1ABC9C"

FIGURES_DIR = os.path.join("reports", "figures")

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 16,
    "axes.labelsize"    : 11,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "figure.dpi"        : 120,
    "figure.facecolor"  : "#FAFBFC",
    "axes.facecolor"    : "#FAFBFC",
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linestyle"    : "--",
})


def _watermark(fig, text="Blood Donation Prediction | Debre Berhan University"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=7, color="#BBBBBB", style="italic",
             transform=fig.transFigure)


class Plotter:
    """
    Beautiful, colorful chart generator for the blood donation project.

    Usage:
        plotter = Plotter(save_dir='reports/figures')
        plotter.correlation_heatmap(df)
    """

    def __init__(self, save_dir: str = FIGURES_DIR):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        _watermark(fig)
        try:
            fig.tight_layout(pad=1.8)
        except Exception:
            pass
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=160, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Plotter] Saved: {path}")

    # ------------------------------------------------------------------ EDA --

    def correlation_heatmap(
        self, df: pd.DataFrame, filename: str = "correlation_heatmap.png"
    ) -> None:
        """Dark-theme Pearson correlation heatmap with vivid diverging colors."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(11, 9))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        cmap = sns.diverging_palette(230, 15, s=85, l=40, as_cmap=True)
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            linewidths=1.2, linecolor="#1E293B", square=True, ax=ax,
            annot_kws={"size": 9, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8, "pad": 0.02},
        )
        ax.set_title("Pearson Correlation Matrix\nBlood Donation Dataset",
                     fontsize=16, fontweight="bold", color="white", pad=20)
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="white")
        self._save(fig, filename)

    def target_distribution(
        self, df: pd.DataFrame,
        target_col: str = "Target",
        filename: str = "target_distribution.png",
    ) -> None:
        """Bar + donut side-by-side target distribution chart."""
        counts = df[target_col].value_counts().sort_index()
        labels = ["Won't Donate (0)", "Will Donate (1)"]
        colors = [PALETTE_RED, PALETTE_BLUE]
        total  = counts.sum()

        fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor("#FAFBFC")

        # Bar
        bars = ax_bar.bar(labels, counts.values, color=colors,
                          edgecolor="white", width=0.52, linewidth=1.5, zorder=3)
        for bar, cnt in zip(bars, counts.values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.015,
                f"{cnt:,}\n({cnt/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
        ax_bar.set_title("Donor Class Distribution", fontweight="bold")
        ax_bar.set_ylabel("Number of Donors")
        ax_bar.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_bar.set_ylim(0, counts.max() * 1.25)

        # Donut
        wedges, texts, autotexts = ax_pie.pie(
            counts.values, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 3},
            pctdistance=0.75, textprops={"fontsize": 11},
        )
        for at in autotexts:
            at.set_fontweight("bold"); at.set_color("white"); at.set_fontsize(13)
        ax_pie.set_title("Class Proportion", fontweight="bold")

        fig.suptitle("Target Variable Distribution — Blood Donation Dataset",
                     fontsize=15, fontweight="bold", y=1.02)
        self._save(fig, filename)

    def feature_distributions(
        self, df: pd.DataFrame, numeric_cols: list = None,
        filename: str = "feature_distributions.png",
    ) -> None:
        """Colorful histogram grid with KDE overlay per feature."""
        numeric_cols = numeric_cols or df.select_dtypes(include=[np.number]).columns.tolist()
        n = len(numeric_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        colors_cycle = [PALETTE_BLUE, PALETTE_RED, PALETTE_PURPLE,
                        PALETTE_GREEN, PALETTE_AMBER, PALETTE_TEAL,
                        "#E67E22", "#2980B9", "#8E44AD"]

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
        fig.patch.set_facecolor("#F8FAFC")
        axes = np.array(axes).ravel()

        for i, col in enumerate(numeric_cols):
            c = colors_cycle[i % len(colors_cycle)]
            data = df[col].dropna()
            axes[i].hist(data, bins=28, color=c, edgecolor="white",
                         alpha=0.78, zorder=3)
            ax2 = axes[i].twinx()
            try:
                data.plot.kde(ax=ax2, color=c, linewidth=2.2)
            except Exception:
                pass
            ax2.set_ylabel(""); ax2.set_yticks([])
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            axes[i].set_title(col, fontsize=11, fontweight="bold", color=c)
            axes[i].set_ylabel("Count", fontsize=9)
            axes[i].set_facecolor("#F8FAFC")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Numeric Feature Distributions  (Histogram + KDE)",
                     fontsize=15, fontweight="bold", y=1.01)
        self._save(fig, filename)

    def boxplots_by_target(
        self, df: pd.DataFrame, target_col: str = "Target",
        numeric_cols: list = None,
        filename: str = "boxplots_by_target.png",
    ) -> None:
        """Colorful split box plots grouped by target class."""
        numeric_cols = numeric_cols or [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
        n = len(numeric_cols)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
        fig.patch.set_facecolor("#F8FAFC")
        axes = np.array(axes).ravel()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(
                x=target_col, y=col, data=df,
                palette={0: PALETTE_RED, 1: PALETTE_BLUE},
                ax=axes[i], linewidth=1.5, fliersize=4,
            )
            axes[i].set_title(col, fontsize=11, fontweight="bold")
            axes[i].set_xlabel("Target", fontsize=9)
            axes[i].set_xticklabels(["Won't Donate (0)", "Will Donate (1)"], fontsize=9)
            axes[i].set_facecolor("#F8FAFC")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        legend_patches = [
            mpatches.Patch(color=PALETTE_RED,  label="Won't Donate (0)"),
            mpatches.Patch(color=PALETTE_BLUE, label="Will Donate (1)"),
        ]
        fig.legend(handles=legend_patches, loc="upper center", ncol=2,
                   fontsize=10, framealpha=0.85, bbox_to_anchor=(0.5, 1.03))
        fig.suptitle("Feature Distributions by Target Class",
                     fontsize=15, fontweight="bold", y=1.06)
        self._save(fig, filename)

    # ---------------------------------------------------- Model Evaluation ---

    def confusion_matrix_plot(
        self, model, X_test: pd.DataFrame, y_test: pd.Series,
        filename: str = "confusion_matrix.png",
    ) -> None:
        """Dark-themed confusion matrix with large TN/FP/FN/TP labels."""
        from sklearn.metrics import confusion_matrix as cm_func
        cm = cm_func(y_test, model.predict(X_test))

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        cmap = LinearSegmentedColormap.from_list(
            "dark_blues", ["#1E293B", "#1D4ED8", "#60A5FA"], N=256)
        ax.imshow(cm, cmap=cmap, aspect="auto")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Won't\nDonate", "Will\nDonate"],
                           fontsize=12, color="white")
        ax.set_yticklabels(["Won't\nDonate", "Will\nDonate"],
                           fontsize=12, color="white")
        ax.set_xlabel("Predicted Label", fontsize=12, color="white")
        ax.set_ylabel("True Label", fontsize=12, color="white")
        ax.tick_params(colors="white")

        labels_map = [["TN", "FP"], ["FN", "TP"]]
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{labels_map[r][c]}\n{cm[r,c]:,}",
                        ha="center", va="center", fontsize=18, fontweight="bold",
                        color="white")

        ax.set_title("Confusion Matrix — Random Forest",
                     fontsize=14, fontweight="bold", color="white", pad=16)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._save(fig, filename)

    def roc_curve_plot(
        self, model, X_test: pd.DataFrame, y_test: pd.Series,
        filename: str = "roc_curve.png",
    ) -> None:
        """Gradient-filled ROC curve with large AUC watermark."""
        from sklearn.metrics import roc_curve, roc_auc_score
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color=PALETTE_BLUE, lw=2.8,
                label=f"Random Forest (AUC = {auc:.3f})", zorder=4)
        ax.fill_between(fpr, tpr, alpha=0.18, color=PALETTE_BLUE)
        ax.plot([0, 1], [0, 1], "k--", lw=1.4,
                label="Random Classifier (AUC = 0.500)")
        ax.text(0.52, 0.08, f"AUC = {auc:.4f}", fontsize=24,
                fontweight="bold", color=PALETTE_BLUE, alpha=0.35,
                transform=ax.transAxes)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve — Random Forest Classifier", fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        self._save(fig, filename)

    def feature_importance_plot(
        self, fi_df: pd.DataFrame, top_n: int = 15,
        filename: str = "feature_importance.png",
    ) -> None:
        """Rainbow-gradient horizontal bar chart of feature importances."""
        top = fi_df.head(top_n).sort_values("importance")
        cmap = plt.cm.RdYlGn
        colors = [cmap(v / top["importance"].max()) for v in top["importance"]]

        fig, ax = plt.subplots(figsize=(9, max(4.5, top_n * 0.45)))
        bars = ax.barh(top["feature"], top["importance"],
                       color=colors, edgecolor="white", linewidth=0.8, height=0.68)
        for bar, val in zip(bars, top["importance"]):
            ax.text(bar.get_width() + top["importance"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left",
                    fontsize=9, fontweight="bold")
        ax.set_xlabel("Gini Importance", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances — Random Forest",
                     fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_xlim(0, top["importance"].max() * 1.18)
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(0, top["importance"].max()))
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="Importance")
        self._save(fig, filename)

    def missing_values_bar(
        self, df: pd.DataFrame, filename: str = "missing_values.png",
    ) -> None:
        """Colorful horizontal bar chart of missing value percentages."""
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]

        if missing_pct.empty:
            print("[Plotter] No missing values to plot.")
            return

        colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(missing_pct)))
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(missing_pct) * 0.55)))
        bars = ax.barh(missing_pct.index, missing_pct.values,
                       color=colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, missing_pct.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left",
                    fontsize=10, fontweight="bold")
        ax.set_xlabel("Missing (%)", fontsize=12)
        ax.set_title("Missing Value Percentage by Column", fontweight="bold")
        ax.set_xlim(0, missing_pct.max() * 1.2)
        self._save(fig, filename)
