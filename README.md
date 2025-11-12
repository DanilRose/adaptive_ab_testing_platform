# Adaptive A/B Testing Platform

Дипломный проект по разработке информационной системы для адаптивного A/B тестирования.

## Структура проекта

adaptive_ab_testing_platform/
├── gan/                          
│   ├── train_gan.py              # Основной скрипт обучения GAN
│   ├── models.py                 # Архитектуры генератора и дискриминатора
│   ├── config.py                 # Гиперпараметры модели
│   └── checkpoints/              # Директория для сохранения моделей
├── traffic_generator/            
│   └── data_generator.py         # Генерация реалистичных пользовательских данных для обучения GAN
├── scripts/                      
│   └── evaluator.py              # Оценка качества синтетических данных
├── requirements.txt
├── README.md

Установка и запуск

1. Клонирование проекта:

```
git clone https://github.com/DanilRose/adaptive_ab_testing_platform.git
cd adaptive_ab_testing_platform
```

2. Установка зависимостей

```
pip install -r requirements.txt
```
3. Запуск проекта

```
cd gan
python train_gan.py
```

