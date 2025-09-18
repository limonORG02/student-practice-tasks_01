"""Модуль для обучения моделей."""

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Обучает модель линейной регрессии и возвращает её."""
    model = LinearRegression()
    model.fit(X, y)
    return model
