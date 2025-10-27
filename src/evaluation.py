from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X, y):
    """Вычисляет метрики качества модели."""
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"[INFO] MSE: {mse:.4f}, R²: {r2:.4f}")
    return {"mse": mse, "r2": r2}
