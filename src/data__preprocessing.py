"""Модуль для загрузки и предобработки данных."""
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Загружает CSV файл в DataFrame."""
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет пропуски и дубликаты."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает базовые статистики по числовым колонкам."""
    return df.describe()
