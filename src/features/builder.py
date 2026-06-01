# =============================================================================
#  File       : src/features/builder.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Creates new features from existing ones and applies
#               feature scaling before model training.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureBuilder:
    """
    Responsible for all feature engineering steps:
      - Deriving new features from existing RFMTC columns
      - Scaling numeric features (StandardScaler or MinMaxScaler)
      - Separating features (X) from the target label (y)

    Usage:
        fb = FeatureBuilder(scale_method='standard')
        X_scaled, y = fb.build_features(df_clean, target_col='Target')
    """

    def __init__(self, scale_method: str = "standard"):
        """
        Parameters
        ----------
        scale_method : str
            'standard' — zero-mean, unit-variance scaling (default)
            'minmax'   — scales each feature to the range [0, 1]
        """
        self.scale_method = scale_method
        self.scaler = None
        self.feature_names: list = []

    # ------------------------------------------------------------------
    # Derive new domain-specific features
    # ------------------------------------------------------------------
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional RFMTC-inspired features that may improve
        the model's ability to separate donors from non-donors.

        New features created
        --------------------
        - donation_density   : Frequency / Time — how often the donor
                               donates per unit of time
        - recency_scaled     : 1 / (Recency + 1) — more recent donors
                               get a higher score
        - high_frequency_flag: 1 if Frequency > median(Frequency), else 0
        """
        df = df.copy()

        # Guard: only add if the required source columns exist
        required = {"Recency", "Frequency", "Monetary", "Time"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            print(f"[FeatureBuilder] Skipping derived features — "
                  f"missing columns: {missing_cols}")
            return df

        # donation_density: how often per month on average
        df["donation_density"] = df["Frequency"] / (df["Time"] + 1e-6)

        # recency_score: inverse recency — fresher donors score higher
        df["recency_score"] = 1.0 / (df["Recency"] + 1)

        # high_frequency_flag: binary flag for frequent donors
        freq_median = df["Frequency"].median()
        df["high_frequency_flag"] = (df["Frequency"] > freq_median).astype(int)

        print(f"[FeatureBuilder] Added 3 derived features: "
              f"donation_density, recency_score, high_frequency_flag")
        return df

    # ------------------------------------------------------------------
    # Scale numeric features
    # ------------------------------------------------------------------
    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale all numeric columns using the selected method.

        Parameters
        ----------
        X   : pd.DataFrame — feature matrix (no target column)
        fit : bool — if True, fit the scaler on X (training set);
                     if False, only transform (test set)

        Returns
        -------
        pd.DataFrame with the same column names but scaled values
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if self.scaler is None or fit:
            if self.scale_method == "standard":
                self.scaler = StandardScaler()
            elif self.scale_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(
                    f"Unknown scale_method '{self.scale_method}'. "
                    "Choose 'standard' or 'minmax'."
                )
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            print(f"[FeatureBuilder] Fitted and applied {self.scale_method} scaler "
                  f"to {len(numeric_cols)} numeric columns.")
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
            print(f"[FeatureBuilder] Applied existing scaler to {len(numeric_cols)} columns.")

        return X

    # ------------------------------------------------------------------
    # Main pipeline entry point
    # ------------------------------------------------------------------
    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str = "Target",
        add_derived: bool = True,
        scale: bool = True,
    ):
        """
        Full feature engineering pipeline:
          1. Optionally add derived features
          2. Split into X (features) and y (target)
          3. Optionally scale numeric columns

        Parameters
        ----------
        df         : cleaned pd.DataFrame
        target_col : name of the target label column
        add_derived: whether to add the derived RFMTC features
        scale      : whether to apply feature scaling

        Returns
        -------
        X : pd.DataFrame — processed feature matrix
        y : pd.Series    — target labels
        """
        df = df.copy()

        # Step 1: Derived features
        if add_derived:
            df = self.add_derived_features(df)

        # Step 2: Separate target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        self.feature_names = X.columns.tolist()

        # Step 3: Scale
        if scale:
            X = self.scale_features(X, fit=True)

        print(f"[FeatureBuilder] Feature matrix shape: {X.shape}")
        print(f"[FeatureBuilder] Target distribution:\n{y.value_counts()}")
        return X, y
