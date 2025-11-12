import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
            
            cohen_d = (real_values.mean() - synth_values.mean()) / np.sqrt(
                (real_values.std() ** 2 + synth_values.std() ** 2) / 2
            )
            
            stats_results[feature] = {
                't_test_pvalue': t_pvalue,
                'ks_test_pvalue': ks_pvalue, 
                'cohen_d': cohen_d,
                'mean_real': real_values.mean(),
                'mean_synth': synth_values.mean(),
                'std_real': real_values.std(),
                'std_synth': synth_values.std()
            }
        
        common_numerical = [f for f in numerical_features if f in self.synthetic_data.columns]
        if len(common_numerical) >= 5:
            real_corr = self.real_data[common_numerical[:5]].corr()
            synth_corr = self.synthetic_data[common_numerical[:5]].corr()
            corr_diff = (real_corr - synth_corr).abs().mean().mean()
        else:
            corr_diff = float('inf')
        
        diversity_score = self._calculate_diversity()
        fid_score = self.calculate_fid_score()
        
        print(f"FID: {fid_score:.2f}")
        print(f"KS среднее: {np.mean([stats.ks_2samp(self.real_data[f], self.synthetic_data[f])[0] for f in numerical_features[:5]]):.4f}")
        print(f"Разница корреляций: {corr_diff:.4f}")
        
        for feature in list(stats_results.keys())[:3]:
            real_mean = stats_results[feature]['mean_real']
            synth_mean = stats_results[feature]['mean_synth']
            diff_pct = abs(real_mean - synth_mean) / real_mean * 100
            print(f"{feature}: {real_mean:.1f} → {synth_mean:.1f} (Δ {diff_pct:.1f}%)")
        
        return {
            'statistical_tests': stats_results,
            'correlation_difference': corr_diff,
            'diversity_score': diversity_score,
            'fid_score': fid_score
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
        
        return diversity

    def calculate_fid_score(self):
        try:
            real_numerical = self.real_data.select_dtypes(include=[np.number])
            synth_numerical = self.synthetic_data.select_dtypes(include=[np.number])
            
            common_columns = list(set(real_numerical.columns) & set(synth_numerical.columns))
            
            if not common_columns:
                return float('inf')
            
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
            
            return fid
            
        except Exception as e:
            return float('inf')

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