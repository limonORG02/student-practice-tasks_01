"""src/data_preprocessing.py
Функции для загрузки, очистки, фичеринга и подготовки данных.
"""
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Загрузить CSV в DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Простая очистка:
      - удаляет полностью пустые строки
      - удаляет дубликаты (опционально)
      - оставляет только числовые колонки и целевую (если она есть)
    """
    df = df.dropna(how="all")
    if drop_duplicates:
        df = df.drop_duplicates()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Базовый feature engineering.
    Добавляет взаимодействие temperature * soil_quality и нормализованную колонку, если есть.
    Модифицируй/расширяй под нужды.
    """
    df = df.copy()
    if "temperature" in df.columns and "soil_quality" in df.columns:
        df["temp_soil_inter"] = df["temperature"] * df["soil_quality"]
    return df


def select_numeric_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in nums if c not in exclude]


def scale_features(
    X: pd.DataFrame, scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Стандартизация (zero mean, unit variance). Возвращает X_scaled и scaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, scaler


def split_features_target(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Разделение на X_train, y_train, X_test, y_test.
    Автоматически выбирает только числовые признаки.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in dataframe columns")

    df = df.copy()
    df = feature_engineering(df)
    numeric = select_numeric_features(df, exclude=[target_col])
    X = df[numeric]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, y_train, X_test, y_test
