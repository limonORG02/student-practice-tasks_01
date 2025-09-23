# Crop Yield Analysis

Аналитический проект для обработки данных сельского хозяйства и прогнозирования урожайности на основе исторических данных.

## Структура проекта
```bash
student-practice-tasks_01/
│── src/ # Исходный код проекта
│ ├── data_preprocessing.py # Загрузка и очистка данных
│ ├── model_training.py # Обучение моделей
│ ├── evaluation.py # Метрики и оценка качества
│ └── init.py
│── data/ # Датасеты
│ └── sample/crop_data.csv
│── test/ # Тесты (pytest)
│ └── test__sample.py
│── requirements.txt # Список зависимостей
│── README.md # Документация проекта

---

## Установка и запуск

### 1. Клонирование репозитория

git clone https://github.com/<org>/student-practice-tasks_01.git
cd student-practice-tasks_01

2. Установка зависимостей

pip install -r requirements.txt

3. Запуск тестов

pytest

Возможности

    Загрузка и предобработка CSV-данных

    Очистка пропусков и аномалий

    Разделение признаков и целевой переменной

    Обучение моделей для прогнозирования урожайности

    Оценка качества моделей по метрике MSE

    Визуализация трендов (matplotlib, seaborn)

Пример использования

from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model_training import train_model
from src.evaluation import evaluate_model

# Загрузка и предобработка
df = load_data("data/sample/crop_data.csv")
df_clean = clean_data(df)
X, y = split_features_target(df_clean, "yield")

# Обучение модели
model = train_model(X, y)

# Оценка
mse = evaluate_model(model, X, y)
print("MSE:", mse)

Технологии

    Python 3.12

    pandas, scikit-learn

    matplotlib, seaborn

    pytest (тестирование)

    black, flake8 (код-стиль)
