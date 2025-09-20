#!/usr/bin/env python3
"""scripts/run_pipeline.py
Простой CLI для запуска пайплайна:
  - загрузка данных
  - очистка/фичеринг/разделение
  - обучение моделей (linear, rf)
  - оценка и сохранение артефактов
"""
import argparse
import json
from pathlib import Path
import logging

import pandas as pd

from src.data_preprocessing import load_data, clean_data, split_features_target, scale_features
from src.model_training import (
    train_linear_regression,
    train_random_forest,
    save_model,
    cross_validate_model,
)
from src.evaluation import evaluate_model, plot_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Run agri-yield pipeline")
    p.add_argument("--data", type=str, default="data/sample/crop_data.csv", help="Path to CSV data")
    p.add_argument("--target", type=str, default="yield", help="Target column name")
    p.add_argument("--models", type=str, default="linear,rf", help="Comma-separated models: linear,rf")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p.add_argument("--outdir", type=str, default="artifacts", help="Output directory for models/reports")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Loading data: %s", args.data)
    df = load_data(args.data)

    logging.info("Cleaning data")
    df = clean_data(df)

    logging.info("Splitting features/target")
    X_train, y_train, X_test, y_test = split_features_target(df, args.target, test_size=args.test_size)

    logging.info("Scaling features")
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _ = scale_features(X_test, scaler=scaler)

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}

    if "linear" in models_to_run:
        logging.info("Training Linear Regression")
        model = train_linear_regression(X_train_scaled, y_train)
        metrics_train = evaluate_model(model, X_train_scaled, y_train)
        metrics_test = evaluate_model(model, X_test_scaled, y_test)
        model_path = outdir / "linear_regression.joblib"
        save_model(model, str(model_path))
        results["linear"] = {"train": metrics_train, "test": metrics_test}
        plot_predictions(y_test, model.predict(X_test_scaled), out_path=str(outdir / "linear_pred.png"))

    if "rf" in models_to_run:
        logging.info("Training Random Forest")
        model = train_random_forest(X_train_scaled, y_train)
        metrics_train = evaluate_model(model, X_train_scaled, y_train)
        metrics_test = evaluate_model(model, X_test_scaled, y_test)
        model_path = outdir / "random_forest.joblib"
        save_model(model, str(model_path))
        results["rf"] = {"train": metrics_train, "test": metrics_test}
        plot_predictions(y_test, model.predict(X_test_scaled), out_path=str(outdir / "rf_pred.png"))

    # кросс-валидация на всех данных для лучшего обзора (опция)
    try:
        logging.info("Cross-validating RandomForest on full data if present")
        if "rf" in models_to_run:
            cv = cross_validate_model(model, pd.concat([X_train_scaled, X_test_scaled]), pd.concat([y_train, y_test]))
            results.setdefault("rf", {})["cv"] = cv
    except Exception:
        logging.exception("CV failed (optional)")

    # сохранить отчет
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(results, indent=2))
    logging.info("Saved report to %s", report_path)
    logging.info("Pipeline finished.")


if __name__ == "__main__":
    main()
