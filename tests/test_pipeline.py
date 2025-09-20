"""tests/test_pipeline.py"""
import os
from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model_training import train_linear_regression
from src.evaluation import evaluate_model

def test_end_to_end_sample():
    path = "data/sample/crop_data.csv"
    assert os.path.exists(path), "sample data not found"
    df = load_data(path)
    df = clean_data(df)
    X_train, y_train, X_test, y_test = split_features_target(df, "yield", test_size=0.4)
    model = train_linear_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    # простая проверка метрики
    assert metrics["mse"] >= 0
