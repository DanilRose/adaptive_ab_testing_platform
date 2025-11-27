import pandas as pd
import numpy as np
import glob
import os
from backend.ab_testing.managers import AdaptiveABTestingPlatform

class ABTestSimulator:
    def __init__(self, platform: AdaptiveABTestingPlatform):
        self.platform = platform
    
    def _find_latest_synthetic_file(self):
        """Находит последний сгенерированный файл с синтетическими данными"""
        # Ищем в корне проекта
        synthetic_files = glob.glob("synthetic_data_*.csv")
        
        if not synthetic_files:
            raise FileNotFoundError(
                "Не найден файл с синтетическими данными в корне проекта. "
                "Сначала сгенерируйте данные через GAN Manager."
            )
        
        # Берем самый новый файл
        latest_file = max(synthetic_files, key=os.path.getctime)
        print(f"📁 Found synthetic data file: {latest_file}")
        return latest_file
    
    def simulate_test(self, test_id: str, synthetic_data_path: str = None, user_count: int = 1000):
        synthetic_data_path = self._find_latest_synthetic_file()
        synthetic_data = pd.read_csv(synthetic_data_path)
        
        print(f"🚀 Starting A/B test simulation for {test_id}")
        
        # Счетчики по вариантам
        variant_counts = {'A': 0, 'B': 0, 'C': 0}
        
        for i, user in synthetic_data.head(user_count).iterrows():
            assignment = self.platform.assign_user_to_test(
                test_id=test_id,
                user_id=str(user['user_id']),
                user_context=user.to_dict()
            )
            
            variant = assignment['variant']
            variant_counts[variant] += 1
            
            # ДЕБАГ: логируем распределение
            if i % 100 == 0:
                print(f"📊 Distribution after {i} users: {variant_counts}")
            
            # Симулируем поведение...
            conversion_rate = self._calculate_conversion_probability(user)
            converted = np.random.random() < conversion_rate
            
            if converted:
                revenue = self._calculate_revenue(user)
                
                # ВАЖНО: Записываем метрики ДО завершения сессии
                primary_metric = self._get_primary_metric(test_id)
                metric_value = revenue if primary_metric == 'revenue' else 1.0
                
                self.platform.record_user_metric(
                    assignment['session_id'], 
                    primary_metric, 
                    metric_value
                )
            
            # Завершаем сессию
            self.platform.complete_user_session(assignment['session_id'])
        
        print(f"✅ A/B test simulation completed for {test_id}")
        print(f"📊 Final distribution: {variant_counts}")

    def _get_primary_metric(self, test_id: str) -> str:
        """Получает основную метрику теста"""
        try:
            # Получаем конфиг теста чтобы узнать основную метрику
            test_config = self.platform.test_manager.test_configs.get(test_id)
            if test_config:
                return test_config.primary_metric
        except:
            pass
        return 'conversion'  # fallback
    
    def _calculate_conversion_probability(self, user: pd.Series) -> float:
        """Расчет вероятности конверсии на основе характеристик пользователя"""
        base_prob = 0.1
        
        # Модификаторы на основе данных пользователя
        if user['user_type'] == 'shopper':
            base_prob += 0.2
        if user['previous_purchases'] > 0:
            base_prob += 0.15
        if user['loyalty_score'] > 0.7:
            base_prob += 0.1
        if user['traffic_source'] == 'direct':
            base_prob += 0.05
            
        # Зависимость от возраста и дохода
        age_factor = max(0, (45 - abs(user['age'] - 35)) / 100)  # пик в 35 лет
        income_factor = min(0.2, user['income'] / 500000)
        
        return min(0.8, base_prob + age_factor + income_factor)
    
    def _calculate_revenue(self, user: pd.Series) -> float:
        """Расчет revenue на основе характеристик пользователя"""
        base_revenue = user['income'] * 0.02  # 2% от дохода
        
        # Модификаторы
        if user['user_type'] == 'shopper':
            base_revenue *= 1.5
        if user['previous_purchases'] > 3:
            base_revenue *= 1.3
        if user['loyalty_score'] > 0.8:
            base_revenue *= 1.2
            
        # Добавляем случайность
        noise = np.random.normal(1.0, 0.2)
        
        return max(10, base_revenue * noise)