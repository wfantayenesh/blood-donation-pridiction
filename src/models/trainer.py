# =============================================================================
#  File       : src/models/trainer.py
#  Project    : Blood Donation Prediction System
#  Authors    : Mihret Alemayehu, Abebech Nega
#  Institution: Debre Berhan University
#  Instructor : Petros Abebe
#  Description: Trains a Random Forest Classifier on the processed blood
#               donation dataset and provides evaluation utilities.
# =============================================================================

import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


class ModelTrainer:
    """
    Wraps the full model training workflow for the blood donation
    prediction task:
      - Train/test splitting (80/20)
      - Random Forest training (n_estimators=100)
      - Cross-validation
      - Evaluation metrics computation
      - Saving/loading model artifacts to the models/ directory

    Usage:
        trainer = ModelTrainer(models_dir='models')
        trainer.fit(X, y)
        trainer.evaluate(X_test, y_test)
        trainer.save_model()
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        test_size: float = 0.20,
        models_dir: str = "models",
    ):
        """
        Parameters
        ----------
        n_estimators  : int    — number of trees in the forest (default 100)
        random_state  : int    — seed for reproducibility (default 42)
        test_size     : float  — fraction held out for testing (default 0.20 = 20%)
        models_dir    : str    — directory where model artifacts are saved
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.test_size = test_size
        self.models_dir = models_dir

        # These are populated after fit()
        self.model: RandomForestClassifier = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.feature_names: list = []
        self.eval_metrics: dict = {}

    # ------------------------------------------------------------------
    # Split data
    # ------------------------------------------------------------------
    def split(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform an 80/20 stratified train/test split.

        Parameters
        ----------
        X : feature matrix
        y : target labels

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,      # preserve class proportions in both splits
        )
        self.feature_names = X.columns.tolist()
        print(f"[ModelTrainer] Train size: {len(self.X_train)} | "
              f"Test size: {len(self.X_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    # Train the Random Forest
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split the data and train a Random Forest Classifier.

        Parameters
        ----------
        X : full feature matrix (will be split internally)
        y : full target series
        """
        # 80/20 split
        self.split(X, y)

        # Initialize the Random Forest with the course-specified hyper-parameters
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,          # use all available CPU cores
            class_weight="balanced",  # handle slight class imbalance
        )

        print(f"[ModelTrainer] Training Random Forest "
              f"(n_estimators={self.n_estimators}, random_state={self.random_state})...")
        self.model.fit(self.X_train, self.y_train)
        train_accuracy = self.model.score(self.X_train, self.y_train)
        print(f"[ModelTrainer] Training accuracy: {train_accuracy:.4f}")

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
        """
        Run k-fold cross-validation and print the mean ± std accuracy.

        Parameters
        ----------
        X  : full feature matrix
        y  : full target series
        cv : number of folds (default 5)
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")
        print(f"[ModelTrainer] {cv}-fold CV Accuracy: "
              f"{scores.mean():.4f} ± {scores.std():.4f}")
        return scores

    # ------------------------------------------------------------------
    # Evaluate on the held-out test set
    # ------------------------------------------------------------------
    def evaluate(self, X_test=None, y_test=None) -> dict:
        """
        Compute evaluation metrics on the test set.

        If X_test / y_test are not provided, uses the split stored in self.

        Returns
        -------
        dict with accuracy, precision, recall, f1, roc_auc, and the
        full sklearn classification_report string.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Call fit() first.")

        X_test = X_test if X_test is not None else self.X_test
        y_test = y_test if y_test is not None else self.y_test

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]  # probability for class 1

        self.eval_metrics = {
            "accuracy"  : round(float(accuracy_score(y_test, y_pred)), 4),
            "precision" : round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall"    : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score"  : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc"   : round(float(roc_auc_score(y_test, y_prob)), 4),
            "report"    : classification_report(y_test, y_pred,
                                                target_names=["Won't Donate", "Will Donate"]),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print("\n===== Evaluation Results =====")
        print(f"  Accuracy  : {self.eval_metrics['accuracy']}")
        print(f"  Precision : {self.eval_metrics['precision']}")
        print(f"  Recall    : {self.eval_metrics['recall']}")
        print(f"  F1 Score  : {self.eval_metrics['f1_score']}")
        print(f"  ROC-AUC   : {self.eval_metrics['roc_auc']}")
        print("\nClassification Report:")
        print(self.eval_metrics["report"])
        print("==============================\n")

        return self.eval_metrics

    # ------------------------------------------------------------------
    # Feature importances
    # ------------------------------------------------------------------
    def feature_importances(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet.")

        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "feature"   : self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return fi_df

    # ------------------------------------------------------------------
    # Save model artifacts
    # ------------------------------------------------------------------
    def save_model(
        self,
        model_filename: str = "model.pkl",
        meta_filename: str = "model_metadata.json",
    ) -> None:
        """
        Serialize the trained model and save metadata to the models/ directory.

        Saves:
          - models/model.pkl          — the trained RandomForestClassifier
          - models/model_metadata.json — accuracy, features, hyper-parameters
        """
        os.makedirs(self.models_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(self.models_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"[ModelTrainer] Model saved to: {model_path}")

        # Save metadata
        metadata = {
            "project"         : "Blood Donation Prediction System",
            "authors"         : ["Mihret Alemayehu", "Abebech Nega"],
            "institution"     : "Debre Berhan University",
            "instructor"      : "Petros Abebe",
            "algorithm"       : "Random Forest Classifier",
            "n_estimators"    : self.n_estimators,
            "random_state"    : self.random_state,
            "test_size"       : self.test_size,
            "train_samples"   : int(len(self.X_train)) if self.X_train is not None else None,
            "test_samples"    : int(len(self.X_test)) if self.X_test is not None else None,
            "features"        : self.feature_names,
            "evaluation"      : {k: v for k, v in self.eval_metrics.items()
                                 if k not in ["report", "confusion_matrix"]},
            "saved_at"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        meta_path = os.path.join(self.models_dir, meta_filename)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[ModelTrainer] Metadata saved to: {meta_path}")

    # ------------------------------------------------------------------
    # Load a saved model
    # ------------------------------------------------------------------
    @staticmethod
    def load_model(model_path: str = "models/model.pkl") -> RandomForestClassifier:
        """
        Load a previously saved model from disk.

        Parameters
        ----------
        model_path : str — path to the .pkl file

        Returns
        -------
        RandomForestClassifier
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[ModelTrainer] Loaded model from: {model_path}")
        return model
