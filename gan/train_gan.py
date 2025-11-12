import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan.config import GANConfig
from gan.models import ProfessionalGAN
from traffic_generator.data_generator import RealisticDataGenerator
from scripts.evaluator import GANEvaluator, TrainingVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def main():
    data_gen = RealisticDataGenerator()
    real_data = data_gen.generate_dataset(50000)
    print(f"   Пользователей: {len(real_data):,}")
    print(f"   Признаков: {len(real_data.columns)}")
    print(f"   total_spent: {real_data['total_spent'].min():.0f} — {real_data['total_spent'].max():.0f}")

    config = GANConfig()
    gan = ProfessionalGAN(config)

    print(f"   Режим: {'WGAN-GP' if config.USE_WGAN_GP else 'Standard GAN'}")
    print(f"   Эпох: {config.EPOCHS}")
    print(f"   Слои G: {config.GENERATOR_LAYERS}")
    print(f"   Слои D: {config.DISCRIMINATOR_LAYERS}")


    start_time = datetime.now()
    gan.train(real_data, epochs=config.EPOCHS)
    end_time = datetime.now()
    print(f"   Время: {end_time - start_time}")

    synthetic_data = gan.generate(100000)
    print(f"   Сгенерировано: {len(synthetic_data):,} пользователей")

    evaluator = GANEvaluator(real_data, synthetic_data, scalers=gan.scalers)
    
    fid_score = evaluator.calculate_fid_score()
    ks_stats = evaluator.calculate_ks_statistics()
    corr_diff = evaluator.calculate_correlation_difference()

    print(f"   FID Score: {fid_score:,.1f}")
    print(f"   KS среднее: {np.mean(list(ks_stats.values())):.4f}")
    print(f"   Разница корреляций: {corr_diff:.4f}")

    stats_ok = 0
    key_features = ['age', 'income', 'previous_purchases', 'session_duration', 'total_spent']
    for feat in key_features:
        if feat in real_data.columns and feat in synthetic_data.columns:
            real_mean = real_data[feat].mean()
            synth_mean = synthetic_data[feat].mean()
            diff = abs(real_mean - synth_mean) / real_mean if real_mean != 0 else 0
            print(f"   {feat}: {real_mean:,.1f} → {synth_mean:,.1f} (Δ {diff:.1%})")
            if diff < 0.2:
                stats_ok += 1
    print(f"   Совпадение средних: {stats_ok}/{len(key_features)}")


    features_to_plot = ['age', 'income', 'previous_purchases', 'total_spent', 'city', 'gender', 'device']
    available = [f for f in features_to_plot if f in real_data.columns and f in synthetic_data.columns]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(available[:8]):
        ax = axes[i]
        if real_data[feat].dtype in ['int64', 'float64']:
            real_data[feat].hist(alpha=0.6, bins=50, label='Реальные', ax=ax, color='skyblue', edgecolor='black')
            synthetic_data[feat].hist(alpha=0.6, bins=50, label='Синтетика', ax=ax, color='salmon', edgecolor='black')
        else:
            real_data[feat].value_counts().plot(kind='bar', alpha=0.6, label='Реальные', ax=ax, color='skyblue')
            synthetic_data[feat].value_counts().plot(kind='bar', alpha=0.6, label='Синтетика', ax=ax, color='salmon')
        ax.set_title(f"{feat}")
        ax.legend()
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig('distributions_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()

    TrainingVisualizer.plot_training_progress(
        g_losses=gan.g_losses,
        d_losses=gan.d_losses,
        gradient_penalties=gan.gradient_penalties if hasattr(gan, 'gradient_penalties') else None,
        wasserstein_distances=gan.wasserstein_distances if hasattr(gan, 'wasserstein_distances') else None,
        save_path='training_progress.png'
    )

    print("\n8. СОХРАНЕНИЕ...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    synthetic_data.to_csv(f'synthetic_users_{timestamp}.csv', index=False)
    evaluator.generate_quality_report(f'quality_report_{timestamp}.html')
    gan._save_checkpoint(f"final_{timestamp}")

    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*50)
    print(f"   Эпох: {len(gan.g_losses)}")
    print(f"   FID: {fid_score:,.1f}")
    if hasattr(gan, 'wasserstein_distances') and gan.wasserstein_distances:
        print(f"   Wasserstein: {gan.wasserstein_distances[-1]:.4f}")
    print(f"   Модель: gan_checkpoint_final_{timestamp}.pth")
    print(f"   Данные: synthetic_users_{timestamp}.csv")
    print("="*50)

if __name__ == "__main__":
    main()