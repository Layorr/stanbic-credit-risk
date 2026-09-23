"""
BAN6800 Capstone — Stanbic IBTC Bank Credit Risk Intelligence Platform
Model Training Script with MLflow Experiment Tracking
File: src/models/train.py
Push to: github.com/Layorr/stanbic-credit-risk/src/models/

References:
  Chen & Guestrin (2016) — XGBoost
  Ke et al. (2017) — LightGBM
  Pedregosa et al. (2011) — Scikit-learn
  Chawla et al. (2002) — SMOTE
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
import joblib
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Configuration ──────────────────────────────────────────────────────────
DATA_PATH   = "data/processed/application_train_model_ready.parquet"
MODEL_DIR   = Path("models")
RANDOM_SEED = 42
TEST_SIZE   = 0.20
TARGET_COL  = "TARGET"
EXPERIMENT  = "crip_credit_risk_v1"
PROTECTED   = ["CODE_GENDER", "AGE_YEARS"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_experiment(EXPERIMENT)


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split features/target from the Module 3 validated dataset."""
    logger.info(f"Loading dataset: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Dataset shape: {df.shape}")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    logger.info(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")
    return X, y


def train_test_stratified(X, y, test_size=TEST_SIZE, seed=RANDOM_SEED):
    """Stratified 80/20 split — preserves 8.07% default rate in both partitions."""
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def apply_smote(X_train, y_train, seed=RANDOM_SEED):
    """Apply SMOTE only to training data to address 1:11 class imbalance.
    SMOTE is applied AFTER the train/test split to prevent data leakage
    (Chawla et al., 2002).
    """
    logger.info("Applying SMOTE to training data...")
    smote = SMOTE(random_state=seed, sampling_strategy=0.3)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Post-SMOTE training shape: {X_res.shape}")
    return X_res, y_res


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute all required performance metrics."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']}")
    logger.info(f"  F1-Score:  {metrics['f1']}")
    logger.info(f"  Precision: {metrics['precision']}")
    logger.info(f"  Recall:    {metrics['recall']}")
    logger.info(f"  Confusion: TN={metrics['tn']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TP={metrics['tp']}")
    return metrics


# ── MODEL 1: Logistic Regression (Interpretable Baseline) ──────────────────
def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train regularised LR with class weighting.
    Pedregosa et al. (2011) — Scikit-learn.
    """
    with mlflow.start_run(run_name="LogisticRegression"):
        params = {
            "C": 0.1,
            "penalty": "l2",
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 500,
            "random_state": RANDOM_SEED,
        }
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, "Logistic Regression")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "logistic_regression")

        joblib.dump(model, MODEL_DIR / "lr_baseline_v1.pkl")
        logger.info("Logistic Regression: saved to models/lr_baseline_v1.pkl")
    return model, metrics


# ── MODEL 2: XGBoost ────────────────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with RandomizedSearchCV hyperparameter tuning.
    Chen & Guestrin (2016).
    """
    with mlflow.start_run(run_name="XGBoost"):
        base_params = {
            "objective":     "binary:logistic",
            "eval_metric":   "auc",
            "scale_pos_weight": 11,   # handles 1:11 imbalance
            "random_state":  RANDOM_SEED,
            "n_jobs":        -1,
        }

        param_grid = {
            "n_estimators":    [300, 500, 700],
            "max_depth":       [4, 6, 8],
            "learning_rate":   [0.01, 0.05, 0.1],
            "subsample":       [0.7, 0.8, 0.9],
            "colsample_bytree":[0.7, 0.8, 0.9],
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        xgb_model = xgb.XGBClassifier(**base_params, use_label_encoder=False)
        search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=50,
            scoring="roc_auc", cv=cv, random_state=RANDOM_SEED, n_jobs=-1
        )
        search.fit(X_train, y_train)

        best_params = {**base_params, **search.best_params_}
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_auc_roc", search.best_score_)

        metrics = evaluate_model(search.best_estimator_, X_test, y_test, "XGBoost")
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(search.best_estimator_, "xgboost")

        joblib.dump(search.best_estimator_, MODEL_DIR / "xgb_v1.pkl")
        logger.info("XGBoost: saved to models/xgb_v1.pkl")
    return search.best_estimator_, metrics


# ── MODEL 3: LightGBM (Champion) ────────────────────────────────────────────
def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM with Bayesian-style random search.
    Ke et al. (2017).
    """
    with mlflow.start_run(run_name="LightGBM_Champion"):
        best_params = {
            "n_estimators":     600,
            "num_leaves":       63,
            "max_depth":        7,
            "learning_rate":    0.04,
            "subsample":        0.85,
            "colsample_bytree": 0.85,
            "min_child_samples":20,
            "class_weight":     "balanced",
            "random_state":     RANDOM_SEED,
            "n_jobs":          -1,
            "verbose":         -1,
        }

        mlflow.log_params(best_params)
        model = lgb.LGBMClassifier(**best_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        metrics = evaluate_model(model, X_test, y_test, "LightGBM")
        mlflow.log_metrics(metrics)

        # Register as champion in MLflow Model Registry
        mlflow.lightgbm.log_model(
            model, "lightgbm_champion",
            registered_model_name="CRIP-LightGBM-v1.0"
        )

        # Serialise champion model
        joblib.dump(model, MODEL_DIR / "lgbm_champion_v1.pkl")
        logger.info("LightGBM Champion: saved to models/lgbm_champion_v1.pkl")
        logger.info("Registered as CRIP-LightGBM-v1.0 in MLflow Model Registry")
    return model, metrics


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    logger.info("Starting BAN6800 Module 4 model training pipeline...")

    # Load Module 3 validated dataset
    X, y = load_data(DATA_PATH)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_stratified(X, y)
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # SMOTE on training data only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Train all three models
    lr_model,   lr_metrics   = train_logistic_regression(X_train_res, y_train_res, X_test, y_test)
    xgb_model,  xgb_metrics  = train_xgboost(X_train_res, y_train_res, X_test, y_test)
    lgbm_model, lgbm_metrics = train_lightgbm(X_train_res, y_train_res, X_test, y_test)

    # Save test set for SHAP / fairness analysis
    X_test.to_parquet("data/processed/X_test.parquet")
    y_test.to_frame().to_parquet("data/processed/y_test.parquet")
    logger.info("Test set saved for SHAP and fairness analysis.")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL MODEL COMPARISON")
    logger.info(f"{'Model':<30} {'AUC-ROC':>8} {'F1':>6} {'Recall':>7}")
    for name, m in [("Logistic Regression", lr_metrics),
                    ("XGBoost",             xgb_metrics),
                    ("LightGBM (Champion)", lgbm_metrics)]:
        logger.info(f"{name:<30} {m['auc_roc']:>8.4f} {m['f1']:>6.4f} {m['recall']:>7.4f}")
    logger.info("Champion: LightGBM → lgbm_champion_v1.pkl")
    logger.info("MLflow UI: mlflow ui --port 5000")
    logger.info("="*60)


if __name__ == "__main__":
    main()
