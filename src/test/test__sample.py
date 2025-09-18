"""Тесты для пайплайна."""

import pandas as pd
from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model_training import train_model
from src.evaluation import evaluate_model


def test_pipeline_with_sample_data():
    # Загружаем sample CSV
    df = load_data("data/sample/crop_data.csv")

    # Предобработка
    df_clean = clean_data(df)
    X, y = split_features_target(df_clean, "yield")

    # Обучение
    model = train_model(X, y)

    # Оценка
    mse = evaluate_model(model, X, y)

    # Проверка, что ошибка существует и >= 0
    assert mse >= 0
