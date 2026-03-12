# Adaptive A/B Testing Platform

Дипломный проект по разработке информационной системы для адаптивного A/B тестирования.

## Структура проекта

```
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
└── README.md
```

# Установка и запуск

1. Клонирование проекта:

```
git clone https://github.com/DanilRose/adaptive_ab_testing_platform.git
cd adaptive_ab_testing_platform
```

2. Установка зависимостей:

```
pip install -r requirements.txt
```

3. Запуск тренировки GAN при необходимости:

```
cd gan
python train_gan.py

```

---

## Оптимизация производительности (v2.0.0)

### Применённые улучшения

1. **Асинхронные API** - все endpoints используют async/await
2. **Кэширование** - часто вызываемые endpoints кэшируются (gan-status, dataset-stats)
3. **Pagination** - поддержка limit/offset для больших датасетов
4. **Индексы БД** - добавлены составные индексы для ускорения запросов
5. **Неблокирующий симулятор** - симуляция A/B тестов не блокирует API

### Быстрый старт с оптимизациями

```bash
# Остановить контейнеры
docker-compose down

# Пересобрать backend с новыми зависимостями
docker-compose build backend

# Применить миграцию БД
docker-compose exec backend python -m backend.database.migration_script

# Запустить приложение
docker-compose up -d
```

### Новые параметры API

```bash
# Pagination для истории данных
GET /api/v1/data/generated-history?limit=50&offset=0&data_type=synthetic

# Response включает метаданные
{
  "items": [...],
  "count": 25,
  "total": 150,
  "has_more": true
}
```

📄 Подробная документация по оптимизации: см. `PERFORMANCE.md` (если существует) или логи миграции.