# Agri Yield Forecast — проект по прогнозированию урожайности

## Краткое описание
Проект посвящён прогнозированию урожайности конкретной культуры на основе данных об окружающей среде и параметрах почвы.  
Цель — сделать воспроизводимый pipeline: **сбор данных → предобработка → обучение моделей → оценка и визуализация**.

**Статус:** начальная инициализация (README + .gitignore + структура репозитория)

---

## Быстрый старт

### Требования
- Git  
- Anaconda (рекомендовано) или Python 3.8+  
- Jupyter / JupyterLab  

### Клонирование и создание окружения (пример)
```bash
git clone <URL_REPO>
cd agri-yield-forecast

# если есть environment.yml
conda env create -f environment.yml
conda activate agri-yield

# либо (если только requirements.txt)
pip install -r requirements.txt

jupyter lab
```

---

## Структура репозитория (предложение)
```
agri-yield-forecast/
├── data/
│   ├── raw/             # исходные (сырые) данные (игнорируются в git)
│   └── processed/       # очищенные/обработанные данные (по необходимости)
├── notebooks/           # Jupyter ноутбуки (исследования)
│   ├── 01_data_cleaning.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/                 # python-модули (data_preprocessing, model_training, evaluation)
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
├── tests/               # тесты pytest
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

---

## Данные
- Помещаем сырые данные в `data/raw/` — эти файлы **не должны попадать в git** (возможно большие/чувствительные).  
- Для этого `data/raw/` включён в `.gitignore`.  
- Обработанные данные и небольшие примеры можно хранить в `data/processed/` или `data/sample/` и отслеживать в репозитории при необходимости.  
- Советуем положить в `data/raw/` файл `.gitkeep`, чтобы каталог существовал в репозитории локально до добавления реальных данных.  

---

## Рабочий процесс и ветки
- `develop` — основная ветка разработки.  
- `main` — ветка релизов/стабильного кода.  
- Feature-ветки: `feature/<описание>`.  

### Пример первого коммита
```bash
git checkout -b develop
git add .
git commit -m "chore(init): initial repo structure, add README and .gitignore"
git push -u origin develop
```

---

## Тестирование
- Запуск: `pytest`  
- Рекомендуется добавить базовые тесты для функций предобработки данных и небольшую тестовую выборку.  

---

## Код-стайл и CI
- Форматирование: `black`  
- Линтер: `flake8` или `ruff`  
- CI (рекомендуется): GitHub Actions для автоматического запуска тестов и линтинга на PR.  

---

## Коммит-месседж и соглашения
- Формат: `type(scope): short description`  
  - Примеры:  
    - `chore(init): ...`  
    - `feat(model): ...`  
    - `fix(data): ...`  
- Пулл-реквесты: PR в `develop`, fast-forward в `main` только после проверки.  

---

## Лицензия
Добавь файл `LICENSE` (например, MIT) по необходимости.  

---

## Следующие шаги (рекомендуемые)
1. Добавить `environment.yml` или `requirements.txt` с версиями библиотек.  
2. Положить небольшой sample-датасет в `data/sample/` для воспроизводимости (или дать ссылку на источник данных).  
3. Создать первый ноутбук `notebooks/01_data_cleaning.ipynb` с базовой предобработкой.  
4. Реализовать функции в `src/data_preprocessing.py` и `src/model_training.py`.  
5. Настроить CI (GitHub Actions) для тестов и линтера.  
```
