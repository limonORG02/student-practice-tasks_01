import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Загружает CSV-файл с данными."""
    try:
        data = pd.read_csv(filepath)
        print(f"[INFO] Данные успешно загружены: {data.shape[0]} строк, {data.shape[1]} столбцов")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {filepath}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает данные: удаляет пропуски и дубликаты."""
    df = df.drop_duplicates()
    df = df.dropna()
    print(f"[INFO] После очистки: {df.shape[0]} строк")
    return df


def split_features_target(df: pd.DataFrame, target: str):
    """Разделяет признаки и целевую переменную."""
    if target not in df.columns:
        raise ValueError(f"Целевая переменная '{target}' не найдена в данных")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
