"""src/model_training.py
Обучение моделей, кросс-валидация, сохранение/загрузка.
"""
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
import joblib
from pathlib import Path


def train_linear_regression(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_random_forest(
    X: pd.DataFrame, y: pd.Series, n_estimators: int = 100, random_state: int = 42
) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    return model


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """Возвращает словарь с средней и std метрики (R2) по кросс-валидации."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    return {"r2_mean": float(scores.mean()), "r2_std": float(scores.std())}


def save_model(model: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
