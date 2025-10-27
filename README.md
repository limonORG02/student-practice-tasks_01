#  Crop Yield Analysis

Аналитический проект для обработки сельскохозяйственных данных и прогнозирования урожайности культур на основе исторических данных.

---

## Описание проекта

Цель проекта — продемонстрировать применение методов **машинного обучения** для анализа факторов окружающей среды (температура, осадки, качество почвы) и прогнозирования урожайности сельхозкультур.

В проекте реализованы:
- загрузка и предварительная обработка данных;
- обучение моделей машинного обучения;
- оценка качества и визуализация результатов.

---

## Структура проекта

```bash
student-practice-tasks_01/
│
├── src/                      # Исходный код
│   ├── data_preprocessing.py # Загрузка и очистка данных
│   ├── model_training.py     # Обучение моделей
│   ├── evaluation.py         # Метрики и оценка качества
│   └── __init__.py
│
├── data/                     # Датасеты
│   └── sample/
│       └── crop_data.csv
│
├── test/                     # Тесты (pytest)
│   └── test_sample.py
│
├── requirements.txt          # Зависимости проекта
└── README.md                 # Документация
```

 Установка и запуск

### 1. Клонирование репозитория

```
git clone git@github.com:limonORG02/student-practice-tasks_01.git
cd student-practice-tasks_01
```

### 2. Создание виртуального окружения

```
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3. Установка зависимостей

```
pip install -r requirements.txt
```

### 4. Запуск тестов

```
./run_tests.sh

```

## Пример использования

```
from src.data_preprocessing import load_data, clean_data, split_features_target
from src.model_training import train_model
from src.evaluation import evaluate_model

# Загрузка и предобработка
df = load_data("data/sample/crop_data.csv")
df_clean = clean_data(df)
X, y = split_features_target(df_clean, target="yield")

# Обучение модели
model = train_model(X, y)

# Оценка
mse = evaluate_model(model, X, y)
print("MSE:", mse)
```

## Используемые технологии

 Python 3.12

 pandas, scikit-learn

 matplotlib, seaborn

 pytest — тестирование

 black, flake8 — стиль и форматирование кода
