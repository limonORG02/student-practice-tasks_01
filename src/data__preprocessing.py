"""Модуль для предобработки данных."""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Загружает данные из CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает данные: удаляет строки с пропущенными значениями."""
    return df.dropna()


def split_features_target(df: pd.DataFrame, target_col: str):
    """Разделяет данные на признаки (X) и целевую переменную (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
