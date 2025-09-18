# Agri Yield Forecast — проект по прогнозированию урожайности

## Краткое описание
Проект посвящён прогнозированию урожайности конкретной культуры на основе данных об окружающей среде и параметрах почвы. Цель — сделать воспроизводимый pipeline: **сбор данных → предобработка → обучение моделей → оценка и визуализация**.

## Статус
Базовая структура проекта создана. Включены рабочие модули для:
- предобработки данных (`src/data_preprocessing.py`),
- обучения моделей (`src/model_training.py`),
- оценки моделей (`src/evaluation.py`).

Добавлен тестовый CSV-файл `data/sample/crop_data.csv`, который используется для проверки пайплайна.

## Быстрый старт

### Требования
- Git
- Anaconda (рекомендовано) или Python 3.8+
- Jupyter / JupyterLab

### Установка

Клонирование репозитория и создание окружения:
```bash
git clone <URL_REPO>
cd agri-yield-forecast

# через conda
tonda env create -f environment.yml
conda activate agri-yield

# либо через pip
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

Запуск JupyterLab:
```bash
jupyter lab
```

### Структура репозитория
```
agri-yield-forecast/
├── data/
│   ├── raw/             # исходные (сырые) данные (игнорируются в git)
│   │   └── .gitkeep
│   ├── processed/       # очищенные/обработанные данные
│   │   └── .gitkeep
│   └── sample/          # пример данных для тестов
│       └── crop_data.csv
├── notebooks/           # Jupyter ноутбуки
│   ├── 01_data_cleaning.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/                 # python-модули
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── tests/               # тесты pytest
│   ├── __init__.py
│   └── test_sample.py
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

### Данные
- **Сырые данные** помещаем в `data/raw/` (игнорируются в git).
- **Обработанные данные** сохраняем в `data/processed/`.
- **Примеры** для тестов и обучения пайплайна — в `data/sample/`.

Пример датасета `data/sample/crop_data.csv` содержит столбцы:
- `temperature` — температура (°C),
- `rain` — осадки (мм),
- `soil_quality` — индекс качества почвы (0–1),
- `yield` — урожайность (т/га).

### Запуск пайплайна

1. Загрузка и очистка данных:
```python
from src.data_preprocessing import load_data, clean_data, split_features_target

df = load_data("data/sample/crop_data.csv")
df_clean = clean_data(df)
X, y = split_features_target(df_clean, "yield")
```

2. Обучение модели:
```python
from src.model_training import train_model

model = train_model(X, y)
```

3. Оценка качества:
```python
from src.evaluation import evaluate_model

mse = evaluate_model(model, X, y)
print("MSE:", mse)
```

### Тестирование
Запуск тестов:
```bash
pytest
```

### Код-стайл
Форматирование и линтеры:
```bash
black src/ tests/
flake8 src/ tests/
```

### Рабочий процесс и ветки
- **develop** — основная ветка разработки.
- **main** — релизы/стабильный код.
- **feature/<описание>** — новые фичи.

### Коммиты
Формат сообщений:
```
type(scope): short description
```
Примеры: `chore(init): ...`, `feat(model): ...`, `fix(data): ...`.

### Лицензия
Добавь файл LICENSE (например, MIT), если требуется.
