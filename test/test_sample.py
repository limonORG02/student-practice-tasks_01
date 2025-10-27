import pandas as pd
from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model_training import train_model
from src.evaluation import evaluate_model


def test_pipeline():
    df = load_data("data/sample/crop_data.csv")
    df_clean = clean_data(df)
    X, y = split_features_target(df_clean, "yield")
    model = train_model(X, y)
    metrics = evaluate_model(model, X, y)

    assert not df_clean.empty
    assert "yield" in df.columns
    assert "mse" in metrics
    assert "r2" in metrics
