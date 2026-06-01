# =============================================================================
#  File       : src/data/loader.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Handles loading raw data from different file formats
#               and performing initial data sanity checks.
# =============================================================================

import os
import pandas as pd


class DataLoader:
    """
    Responsible for loading datasets from various file formats
    (CSV, Excel, Parquet) into pandas DataFrames.

    Usage:
        loader = DataLoader(data_dir='data/raw')
        df = loader.load_csv('blood_donation.csv')
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Parameters
        ----------
        data_dir : str
            Base directory where raw data files are stored.
        """
        self.data_dir = data_dir

    def _build_path(self, filename: str) -> str:
        """Build the full path to the data file."""
        return os.path.join(self.data_dir, filename)

    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame.

        Parameters
        ----------
        filename : str
            Name of the CSV file (e.g., 'blood_donation.csv').
        **kwargs  : additional arguments forwarded to pd.read_csv

        Returns
        -------
        pd.DataFrame
        """
        filepath = self._build_path(filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV not found at: {filepath}")

        df = pd.read_csv(filepath, **kwargs)
        print(f"[DataLoader] Loaded '{filename}': {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def load_excel(self, filename: str, sheet_name: int = 0, **kwargs) -> pd.DataFrame:
        """
        Load an Excel file into a DataFrame.

        Parameters
        ----------
        filename   : str
        sheet_name : int or str, default 0
        """
        filepath = self._build_path(filename)
        df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
        print(f"[DataLoader] Loaded '{filename}' (sheet {sheet_name}): {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """
        Load a Parquet file into a DataFrame.

        Parameters
        ----------
        filename : str
        """
        filepath = self._build_path(filename)
        df = pd.read_parquet(filepath, **kwargs)
        print(f"[DataLoader] Loaded '{filename}': {df.shape[0]} rows × {df.shape[1]} columns")
        return df

    def quick_summary(self, df: pd.DataFrame) -> None:
        """
        Print a quick overview of the loaded DataFrame:
        shape, column dtypes, missing value counts, and
        the first 5 rows.
        """
        print("\n===== Dataset Summary =====")
        print(f"  Shape     : {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Duplicates: {df.duplicated().sum()}")
        print(f"  Total NaN : {df.isnull().sum().sum()}")
        print("\n--- Column Types ---")
        print(df.dtypes)
        print("\n--- Missing Values per Column ---")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  No missing values found.")
        else:
            print(missing)
        print("\n--- First 5 Rows ---")
        print(df.head())
        print("===========================\n")
