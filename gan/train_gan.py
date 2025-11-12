import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan.config import GANConfig
from gan.models import ProfessionalGAN
from traffic_generator.data_generator import RealisticDataGenerator
from scripts.evaluator import GANEvaluator
import pandas as pd
import numpy as np
from datetime import datetime

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    synthetic_data.to_csv(f'synthetic_users_{timestamp}.csv', index=False)
    gan._save_checkpoint(f"final_{timestamp}")

    print(f"   Эпох: {len(gan.g_losses)}")
    print(f"   FID: {fid_score:,.1f}")
    if hasattr(gan, 'wasserstein_distances') and gan.wasserstein_distances:
        print(f"   Wasserstein: {gan.wasserstein_distances[-1]:.4f}")
    print(f"   Модель: gan_checkpoint_final_{timestamp}.pth")
    print(f"   Данные: synthetic_users_{timestamp}.csv")

if __name__ == "__main__":
    main()