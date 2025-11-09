import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan.config import GANConfig
from gan.models import ProfessionalGAN
from traffic_generator.data_generator import RealisticDataGenerator
from scripts.evaluator import GANEvaluator, TrainingVisualizer
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("🚀 ЗАПУСК УЛУЧШЕННОГО GAN С ВАЛИДАЦИЕЙ")
    
    # 1. Генерация данных
    print("1. 📊 Генерация тренировочных данных...")
    data_gen = RealisticDataGenerator()
    real_data = data_gen.generate_dataset(50000)
    print(f"   Сгенерировано: {len(real_data):,} пользователей, {len(real_data.columns)} признаков")
    
    # 2. Обучение GAN с валидацией
    print("2. 🧠 Запуск обучения GAN с валидацией...")
    config = GANConfig()
    gan = ProfessionalGAN(config)
    
    gan.train(real_data, epochs=1000)
    
    # 3. Финальная оценка
    print("3. 📈 Финальная оценка качества...")
    synthetic_data = gan.generate(20000)
    
    evaluator = GANEvaluator(real_data, synthetic_data)
    evaluation_results = evaluator.evaluate_quality()
    
    # 4. Визуализация
    print("4. 📊 Создание визуализаций...")
    features_to_plot = ['age', 'income', 'previous_purchases', 'city', 'gender']
    available_features = [f for f in features_to_plot if f in real_data.columns and f in synthetic_data.columns]
    evaluator.plot_distributions(available_features)
    
    # 5. Прогресс обучения
    TrainingVisualizer.plot_training_progress(
        gan.g_losses, 
        gan.d_losses, 
        gan.gradient_penalties if hasattr(gan, 'gradient_penalties') else None,
        gan.wasserstein_distances if hasattr(gan, 'wasserstein_distances') else None
    )
    
    # 6. Сохранение
    synthetic_data.to_csv('synthetic_users_improved.csv', index=False)
    evaluator.generate_quality_report()
    
    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("📊 Качество улучшено за счет:")
    print("   - Валидации на каждой 200-й эпохе")
    print("   - Early stopping по FID score")
    print("   - Оптимизированных гиперпараметров")
    print("   - Исправленной обработки категориальных признаков")

if __name__ == "__main__":
    main()