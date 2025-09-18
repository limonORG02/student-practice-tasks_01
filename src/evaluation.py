"""Модуль для оценки моделей."""

import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Вычисляет среднеквадратичную ошибку модели."""
    predictions = model.predict(X_test)
    return mean_squared_error(y_test, predictions)
