import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math

warnings.filterwarnings('ignore')

from backend.microservices.shared.utils import sanitize_float


class GANEvaluator:
    def __init__(self, real_data, synthetic_data, scalers=None):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.scalers = scalers 
    
    def evaluate_quality(self):
        numerical_features = self.real_data.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f in self.synthetic_data.columns]
        
        stats_results = {}
        for feature in numerical_features[:10]:
            real_values = self.real_data[feature]
            synth_values = self.synthetic_data[feature]
            
            t_stat, t_pvalue = stats.ttest_ind(real_values, synth_values)
            ks_stat, ks_pvalue = stats.ks_2samp(real_values, synth_values)
            
            pooled_std = np.sqrt(
                (real_values.std() ** 2 + synth_values.std() ** 2) / 2
            )
            if pooled_std != 0:
                cohen_d = (real_values.mean() - synth_values.mean()) / pooled_std
            else:
                cohen_d = 0.0
            
            stats_results[feature] = {
                't_test_pvalue': sanitize_float(float(t_pvalue)),
                'ks_test_pvalue': sanitize_float(float(ks_pvalue)),
                'cohen_d': sanitize_float(float(cohen_d)),
                'mean_real': sanitize_float(float(real_values.mean())),
                'mean_synth': sanitize_float(float(synth_values.mean())),
                'std_real': sanitize_float(float(real_values.std())),
                'std_synth': sanitize_float(float(synth_values.std()))
            }
        
        common_numerical = [f for f in numerical_features if f in self.synthetic_data.columns]
        if len(common_numerical) >= 5:
            real_corr = self.real_data[common_numerical[:5]].corr()
            synth_corr = self.synthetic_data[common_numerical[:5]].corr()
            corr_diff_raw = (real_corr - synth_corr).abs().mean().mean()
            corr_diff = sanitize_float(float(corr_diff_raw))
        else:
            corr_diff = None
        
        diversity_score = self._calculate_diversity()
        fid_score = self.calculate_fid_score()
        wasserstein_result = self.calculate_wasserstein_distances()

        if fid_score is not None:
            print(f"FID: {fid_score:.2f}")
        else:
            print("FID: N/A")
        if len(numerical_features) > 0:
            ks_mean_raw = np.mean([stats.ks_2samp(self.real_data[f], self.synthetic_data[f])[0] for f in numerical_features[:5]])
            print(f"KS среднее: {ks_mean_raw:.4f}")
        if corr_diff is not None:
            print(f"Разница корреляций: {corr_diff:.4f}")
        mean_wd = wasserstein_result.get("mean_wd")
        if mean_wd is not None:
            print(f"Wasserstein Distance (среднее): {mean_wd:.4f}")
        wd_quality = wasserstein_result.get("quality_score")
        if wd_quality is not None:
            print(f"WD Quality Score: {wd_quality:.4f}")

        for feature in list(stats_results.keys())[:3]:
            real_mean = stats_results[feature]['mean_real']
            synth_mean = stats_results[feature]['mean_synth']
            if real_mean is not None and synth_mean is not None and real_mean != 0:
                diff_pct = abs(real_mean - synth_mean) / real_mean * 100
            else:
                diff_pct = 0.0
            print(f"{feature}: {real_mean} → {synth_mean} (Δ {diff_pct:.1f}%)")

        return {
            'statistical_tests': stats_results,
            'correlation_difference': corr_diff,
            'diversity_score': sanitize_float(diversity_score),
            'fid_score': fid_score,
            'wasserstein': wasserstein_result,
        }

    def _calculate_diversity(self):
        numerical_data = self.synthetic_data.select_dtypes(include=[np.number])
        if len(numerical_data.columns) == 0:
            return 0.0
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numerical_data)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        x_range = pca_result[:, 0].max() - pca_result[:, 0].min()
        y_range = pca_result[:, 1].max() - pca_result[:, 1].min()
        diversity = x_range * y_range
        
        return float(diversity) if not (math.isnan(float(diversity)) or math.isinf(float(diversity))) else 0.0

    def calculate_wasserstein_distances(self) -> dict:
        """
        Вычисляет Wasserstein Distance (Earth Mover's Distance) между
        реальными и синтетическими распределениями для каждого числового признака.

        Wasserstein Distance — стандартная метрика качества GAN:
        - WD = 0: распределения идентичны
        - Чем меньше WD, тем лучше GAN воспроизводит реальные данные
        - Устойчива к выбросам и не требует одинакового числа наблюдений

        Returns:
            dict с полями:
                wasserstein_distances: {feature: distance}
                mean_wd: среднее WD по всем признакам
                max_wd: максимальное WD (наиболее проблемный признак)
                min_wd: минимальное WD (лучше всего воспроизведённый признак)
                quality_score: нормализованная оценка [0, 1] (1 = идеально)
        """
        real_numerical  = self.real_data.select_dtypes(include=[np.number])
        synth_numerical = self.synthetic_data.select_dtypes(include=[np.number])

        common_columns = [
            col for col in real_numerical.columns
            if col in synth_numerical.columns
        ]

        if not common_columns:
            return {
                "wasserstein_distances": {},
                "mean_wd": None,
                "max_wd": None,
                "min_wd": None,
                "quality_score": None,
                "error": "No common numerical columns found",
            }

        distances: dict = {}
        for col in common_columns:
            real_vals  = real_numerical[col].dropna().values
            synth_vals = synth_numerical[col].dropna().values

            if len(real_vals) == 0 or len(synth_vals) == 0:
                continue

            try:
                wd = wasserstein_distance(real_vals, synth_vals)
                distances[col] = sanitize_float(float(wd))
            except Exception:
                pass

        if not distances:
            return {
                "wasserstein_distances": {},
                "mean_wd": None,
                "max_wd": None,
                "min_wd": None,
                "quality_score": None,
            }

        wd_values = [v for v in distances.values() if v is not None]
        mean_wd = float(np.mean(wd_values))
        max_wd  = float(np.max(wd_values))
        min_wd  = float(np.min(wd_values))

        # Нормализованная оценка качества: используем экспоненциальное затухание.
        # При mean_wd = 0 → quality_score = 1.0 (идеально).
        # При mean_wd = 1 → quality_score ≈ 0.37.
        # Масштаб нормировки зависит от диапазона данных; используем медиану std реальных данных.
        real_stds = real_numerical[list(distances.keys())].std().dropna()
        scale = float(real_stds.median()) if len(real_stds) > 0 and real_stds.median() > 0 else 1.0
        quality_score = float(np.exp(-mean_wd / scale))
        quality_score = sanitize_float(min(1.0, max(0.0, quality_score)))

        return {
            "wasserstein_distances": distances,
            "mean_wd": sanitize_float(mean_wd),
            "max_wd": sanitize_float(max_wd),
            "min_wd": sanitize_float(min_wd),
            "quality_score": quality_score,
            "worst_feature": max(distances, key=lambda k: distances[k] or 0),
            "best_feature":  min(distances, key=lambda k: distances[k] or float("inf")),
        }

    def calculate_fid_score(self):
        try:
            real_numerical = self.real_data.select_dtypes(include=[np.number])
            synth_numerical = self.synthetic_data.select_dtypes(include=[np.number])
            
            common_columns = list(set(real_numerical.columns) & set(synth_numerical.columns))
            
            if not common_columns:
                return None
            
            if self.scalers:
                real_scaled = real_numerical[common_columns].copy()
                synth_scaled = synth_numerical[common_columns].copy()
                
                for col in common_columns:
                    if col in self.scalers:
                        scaler = self.scalers[col]
                        real_scaled[col] = scaler.transform(real_numerical[[col]])
                        synth_scaled[col] = scaler.transform(synth_numerical[[col]])
                
                real_samples = real_scaled.values
                synth_samples = synth_scaled.values
            else:
                real_samples = real_numerical[common_columns].values
                synth_samples = synth_numerical[common_columns].values
            
            mu1, sigma1 = np.mean(real_samples, axis=0), np.cov(real_samples, rowvar=False)
            mu2, sigma2 = np.mean(synth_samples, axis=0), np.cov(synth_samples, rowvar=False)
            
            diff = mu1 - mu2
            covmean = sqrtm(sigma1.dot(sigma2))
            
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            
            return sanitize_float(float(fid))
            
        except Exception as e:
            return None

    def plot_distributions(self, features, n_cols=3, save_path='distribution_comparison.png'):
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if feature in self.real_data.columns and feature in self.synthetic_data.columns:
                if self.real_data[feature].dtype in ['int64', 'float64']:
                    axes[i].hist(self.real_data[feature], bins=30, alpha=0.7, label='Real', density=True, color='blue')
                    axes[i].hist(self.synthetic_data[feature], bins=30, alpha=0.7, label='Synthetic', density=True, color='orange')
                    axes[i].set_title(f'{feature}')
                    axes[i].legend()
        
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class TrainingVisualizer:
    @staticmethod
    def plot_training_progress(g_losses, d_losses, gradient_penalties=None, wasserstein_distances=None, save_path='training_progress.png'):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(g_losses, label='Generator Loss', alpha=0.7)
        plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
        plt.title('Training Losses')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if gradient_penalties:
            plt.subplot(1, 3, 2)
            plt.plot(gradient_penalties, label='Gradient Penalty', alpha=0.7, color='green')
            plt.title('Gradient Penalty')
            plt.xlabel('Iteration')
            plt.ylabel('GP Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if wasserstein_distances:
            plt.subplot(1, 3, 3)
            plt.plot([abs(w) for w in wasserstein_distances], label='Wasserstein Distance', alpha=0.7, color='red')
            plt.title('Wasserstein Distance')
            plt.xlabel('Iteration')
            plt.ylabel('Distance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
