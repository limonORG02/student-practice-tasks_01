"""src/evaluation.py
Метрики и визуализация предсказаний.
"""
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    preds = model.predict(X)
    return compute_metrics(y, pd.Series(preds, index=y.index))


def plot_predictions(
    y_true: pd.Series, y_pred: pd.Series, out_path: str = "reports/pred_vs_true.png", show: bool = False
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    plt.plot(lims, lims, "--", linewidth=1, label="y = pred")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("Predicted vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()
