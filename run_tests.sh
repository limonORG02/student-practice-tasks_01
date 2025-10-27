#!/bin/bash
# Скрипт для запуска pytest с правильным PYTHONPATH и venv

# Активация venv (путь к твоей виртуальной среде)
source ./venv/bin/activate

# Установка PYTHONPATH на src
export PYTHONPATH=$(pwd)/src

# Запуск pytest с аргументами
python -m pytest "$@"
