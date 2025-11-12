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
        print("\n" + "="*60)
        print("РАСШИРЕННАЯ ОЦЕНКА КАЧЕСТВА GAN")
        print("="*60)
        
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
        
        print(f"\n📊 СТАТИСТИЧЕСКИЕ ТЕСТЫ (первые 5 признаков):")
        for i, (feature, results) in enumerate(list(stats_results.items())[:5]):
            status_t = "✅" if results['t_test_pvalue'] > 0.05 else "❌"
            status_ks = "✅" if results['ks_test_pvalue'] > 0.05 else "❌" 
            status_d = "✅" if abs(results['cohen_d']) < 0.2 else "❌"
            
            print(f"  {feature}:")
            print(f"    T-тест: p={results['t_test_pvalue']:.4f} {status_t}")
            print(f"    KS-тест: p={results['ks_test_pvalue']:.4f} {status_ks}")
            print(f"    Cohen's d: {results['cohen_d']:.3f} {status_d}")
        
        print(f"\n📈 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ:")
        print(f"  Средняя разница в корреляциях: {corr_diff:.4f}")
        print(f"  {'✅ Отлично' if corr_diff < 0.1 else '⚠️  Хорошо' if corr_diff < 0.2 else '❌ Требует улучшения'}")
        
        print(f"\n🎯 ОБЩАЯ ОЦЕНКА:")
        print(f"  Разнообразие данных: {diversity_score:.3f}")
        print(f"  FID Score: {fid_score:.2f}")
        print(f"  Размер реальных данных: {len(self.real_data):,}")
        print(f"  Размер синтетических данных: {len(self.synthetic_data):,}")
        print(f"  Общих признаков: {len(common_numerical)}")
        
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
            print(f"Ошибка при вычислении FID: {e}")
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
                    axes[i].set_title(f'Distribution of {feature}')
                    axes[i].legend()
                else:
                    real_counts = self.real_data[feature].value_counts().head(10)
                    synth_counts = self.synthetic_data[feature].value_counts()
                    all_categories = list(real_counts.index)
                    for cat in synth_counts.index:
                        if cat not in all_categories and len(all_categories) < 10:
                            all_categories.append(cat)
                    
                    real_values = [real_counts.get(cat, 0) for cat in all_categories]
                    synth_values = [synth_counts.get(cat, 0) for cat in all_categories]
                    
                    x = np.arange(len(all_categories))
                    width = 0.35
                    
                    axes[i].bar(x - width/2, real_values, width, label='Real', alpha=0.7, color='blue')
                    axes[i].bar(x + width/2, synth_values, width, label='Synthetic', alpha=0.7, color='orange')
                    axes[i].set_xticks(x)
                    axes[i].set_xticklabels(all_categories, rotation=45, ha='right')
                    axes[i].set_title(f'Distribution of {feature}')
                    axes[i].legend()
        
        # Скрываем пустые subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def calculate_data_quality_metrics(self):
        metrics = {}
        
        common_features = set(self.real_data.columns) & set(self.synthetic_data.columns)
        metrics['feature_overlap'] = len(common_features) / len(self.real_data.columns)
        
        numerical_features = self.real_data.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f in common_features]
        
        statistical_metrics = []
        for feature in numerical_features[:10]:
            real_values = self.real_data[feature]
            synth_values = self.synthetic_data[feature]
            
            ks_stat, ks_pvalue = stats.ks_2samp(real_values, synth_values)
            mape = np.mean(np.abs((real_values.mean() - synth_values.mean()) / real_values.mean()))
            
            statistical_metrics.append({
                'feature': feature,
                'ks_pvalue': ks_pvalue,
                'mape': mape
            })
        
        metrics['statistical_metrics'] = statistical_metrics
        metrics['avg_ks_pvalue'] = np.mean([m['ks_pvalue'] for m in statistical_metrics])
        metrics['avg_mape'] = np.mean([m['mape'] for m in statistical_metrics])
        
        return metrics

    def generate_quality_report(self, save_path='gan_quality_report.txt'):
        report = self.evaluate_quality()
        metrics = self.calculate_data_quality_metrics()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ О КАЧЕСТВЕ GAN\n")
            f.write("="*50 + "\n\n")
            
            f.write("ОБЩАЯ ИНФОРМАЦИЯ:\n")
            f.write(f"Реальные данные: {len(self.real_data):,} записей\n")
            f.write(f"Синтетические данные: {len(self.synthetic_data):,} записей\n")
            f.write(f"Общие признаки: {len(set(self.real_data.columns) & set(self.synthetic_data.columns))}\n\n")
            
            f.write("СТАТИСТИЧЕСКИЕ МЕТРИКИ:\n")
            f.write(f"Средний p-value KS теста: {metrics['avg_ks_pvalue']:.4f}\n")
            f.write(f"Средняя MAPE: {metrics['avg_mape']:.4f}\n")
            f.write(f"FID Score: {report['fid_score']:.2f}\n")
            f.write(f"Разнообразие: {report['diversity_score']:.3f}\n\n")
            
            f.write("ОЦЕНКА КАЧЕСТВА:\n")
            if report['fid_score'] < 100 and metrics['avg_ks_pvalue'] > 0.05:
                f.write("✅ ОТЛИЧНО - данные высокого качества\n")
            elif report['fid_score'] < 500 and metrics['avg_ks_pvalue'] > 0.01:
                f.write("⚠️  ХОРОШО - данные приемлемого качества\n")
            else:
                f.write("❌ ТРЕБУЕТ УЛУЧШЕНИЯ - низкое качество данных\n")
        
        print(f"Отчет сохранен в {save_path}")

    def calculate_ks_statistics(self):
        numerical_features = self.real_data.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f in self.synthetic_data.columns]
        
        ks_stats = {}
        for feature in numerical_features:
            real_values = self.real_data[feature]
            synth_values = self.synthetic_data[feature]
            ks_stat, _ = stats.ks_2samp(real_values, synth_values)
            ks_stats[feature] = ks_stat
        
        return ks_stats

    def calculate_correlation_difference(self):
        numerical_features = self.real_data.select_dtypes(include=[np.number]).columns
        numerical_features = [f for f in numerical_features if f in self.synthetic_data.columns]
        
        if len(numerical_features) >= 5:
            real_corr = self.real_data[numerical_features[:5]].corr()
            synth_corr = self.synthetic_data[numerical_features[:5]].corr()
            corr_diff = (real_corr - synth_corr).abs().mean().mean()
            return corr_diff
        else:
            return float('inf')

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