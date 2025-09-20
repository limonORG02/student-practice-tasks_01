# Agri Yield Forecast

Проект для прогнозирования урожайности конкретной культуры на основе данных об окружающей среде и параметрах почвы. Цель — воспроизводимый pipeline: загрузка данных → предобработка → обучение моделей → оценка → визуализация.

## Быстрый старт

### Требования
- Python 3.10+ (рекомендуется через venv или conda)
- Jupyter / JupyterLab
- Git

### Клонирование и установка зависимостей
```bash
git clone <URL_REPO>
cd agri-yield-forecast

# Через pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Через conda (если используешь environment.yml)
conda env create -f environment.yml
conda activate agri-yield
