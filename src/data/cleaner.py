# =============================================================================
#  File       : src/data/cleaner.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Cleans and preprocesses raw data — handles missing values,
#               duplicates, and type conversions to produce a clean dataset
#               ready for feature engineering.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class DataCleaner:
    """
    Handles all data cleaning steps:
      - Standardizing missing value representations
      - Removing duplicate records
      - Imputing missing values (KNN by default)
      - Encoding categorical columns
      - Dropping irrelevant columns

    Usage:
        cleaner = DataCleaner(n_neighbors=5)
        df_clean = cleaner.clean(df_raw, target_col='Target')
    """

    def __init__(self, n_neighbors: int = 5):
        """
        Parameters
        ----------
        n_neighbors : int
            Number of neighbors used by the KNN imputer. Default is 5
            as specified in the course guidelines.
        """
        self.n_neighbors = n_neighbors
        self.knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        self.label_encoders: dict = {}   # stores fitted LabelEncoders per column

    # ------------------------------------------------------------------
    # Step 1: Standardize missing value markers
    # ------------------------------------------------------------------
    def standardize_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace common placeholders ('NA', 'N/A', '--', empty string, etc.)
        with np.nan so that pandas and sklearn can recognize them as missing.
        """
        placeholders = ["", " ", "NA", "N/A", "nan", "NaN", "--", "-", "?", "none", "None"]
        df = df.replace(placeholders, np.nan)
        return df

    # ------------------------------------------------------------------
    # Step 2: Remove exact duplicate rows
    # ------------------------------------------------------------------
    def remove_duplicates(self, df: pd.DataFrame, reset_index: bool = True) -> pd.DataFrame:
        """
        Drop exact duplicate rows and optionally reset the integer index.

        Parameters
        ----------
        df          : pd.DataFrame
        reset_index : bool, default True
        """
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed:
            print(f"[DataCleaner] Removed {n_removed} duplicate rows.")
        else:
            print("[DataCleaner] No duplicate rows found.")

        if reset_index:
            df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Step 3: KNN imputation for numeric columns
    # ------------------------------------------------------------------
    def impute_knn(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """
        Impute missing values in numeric columns using K-Nearest Neighbors.

        Non-numeric (categorical) columns and any columns in `exclude_cols`
        are left untouched.

        Parameters
        ----------
        df           : pd.DataFrame
        exclude_cols : list of column names to skip during imputation
                       (e.g., the target column)
        """
        exclude_cols = exclude_cols or []

        # Identify numeric columns that are eligible for imputation
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        missing_before = df[numeric_cols].isnull().sum().sum()
        if missing_before == 0:
            print("[DataCleaner] No missing values to impute in numeric columns.")
            return df

        # Fit and transform only the numeric feature columns
        df[numeric_cols] = self.knn_imputer.fit_transform(df[numeric_cols])

        missing_after = df[numeric_cols].isnull().sum().sum()
        print(f"[DataCleaner] KNN imputation done. "
              f"Missing values before: {missing_before}, after: {missing_after}")
        return df

    # ------------------------------------------------------------------
    # Step 4: Encode categorical columns
    # ------------------------------------------------------------------
    def encode_categoricals(self, df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
        """
        Label-encode all object/categorical columns (except those in exclude_cols).

        A LabelEncoder is stored in self.label_encoders[col] so it can be
        inverse-transformed later if needed.

        Parameters
        ----------
        df           : pd.DataFrame
        exclude_cols : list of column names to skip
        """
        exclude_cols = exclude_cols or []
        cat_cols = [
            col for col in df.select_dtypes(include=["object", "category"]).columns
            if col not in exclude_cols
        ]

        for col in cat_cols:
            le = LabelEncoder()
            # fill any remaining categorical NaN with the string 'Unknown'
            df[col] = df[col].fillna("Unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"[DataCleaner] Encoded '{col}' — classes: {list(le.classes_)}")

        return df

    # ------------------------------------------------------------------
    # Step 5: Drop columns that provide no predictive value
    # ------------------------------------------------------------------
    def drop_irrelevant_columns(self, df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Parameters
        ----------
        df           : pd.DataFrame
        cols_to_drop : list of column names to remove
        """
        existing = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=existing)
        print(f"[DataCleaner] Dropped columns: {existing}")
        return df

    # ------------------------------------------------------------------
    # Full cleaning pipeline
    # ------------------------------------------------------------------
    def clean(
        self,
        df: pd.DataFrame,
        target_col: str = "Target",
        id_cols: list = None,
    ) -> pd.DataFrame:
        """
        Run the full cleaning pipeline in order:
          1. Standardize missing value markers
          2. Remove duplicate rows
          3. Impute missing numeric values with KNN (k=5)
          4. Encode remaining categorical columns

        Parameters
        ----------
        df         : raw pd.DataFrame
        target_col : name of the label/target column (excluded from imputation)
        id_cols    : optional list of ID columns to drop before imputation

        Returns
        -------
        pd.DataFrame — clean, imputation-complete dataset
        """
        print("\n[DataCleaner] Starting cleaning pipeline...")
        df = df.copy()

        # Step 1
        df = self.standardize_missing(df)

        # Step 2
        df = self.remove_duplicates(df)

        # Step 3: Drop pure-ID columns before imputing
        if id_cols:
            df = self.drop_irrelevant_columns(df, id_cols)

        # KNN impute (skip the target column)
        df = self.impute_knn(df, exclude_cols=[target_col])

        # Step 4: Encode any remaining categorical columns (excluding target)
        df = self.encode_categoricals(df, exclude_cols=[target_col])

        print(f"[DataCleaner] Pipeline complete. Final shape: {df.shape}\n")
        return df
