import pandas as pd
import numpy as np
from backend.ab_testing.managers import AdaptiveABTestingPlatform
from backend.database.session import SessionLocal
from backend.database import crud

class ABTestSimulator:
    def __init__(self, platform: AdaptiveABTestingPlatform):
        self.platform = platform
    
    def _load_latest_synthetic_data(self) -> pd.DataFrame:
        with SessionLocal() as db:
            latest = crud.get_latest_generated_data_by_type(db, "synthetic")

        if latest is None:
            raise ValueError("Не найдены синтетические данные в БД. Сначала выполните генерацию в GAN Manager.")

        metadata = latest.extra_metadata or {}
        records = metadata.get("records")
        if not records:
            records = latest.preview_json or []

        if not records:
            raise ValueError("В БД нет записей synthetic data для симуляции.")

        return pd.DataFrame(records)
    
    def simulate_test(self, test_id: str, synthetic_data_path: str = None, user_count: int = 1000):
        synthetic_data = self._load_latest_synthetic_data()
        
        print(f" Starting A/B test simulation for {test_id}")
        
        variant_counts = {}
        
        for i, user in synthetic_data.head(user_count).iterrows():
            assignment = self.platform.assign_user_to_test(
                test_id=test_id,
                user_id=str(user['user_id']),
                user_context=user.to_dict()
            )
            
            variant = assignment['variant']
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
            
            if i % 100 == 0:
                print(f"📊 Distribution after {i} users: {variant_counts}")
            
            conversion_rate = self._calculate_conversion_probability(user)
            converted = np.random.random() < conversion_rate
            
            if converted:
                revenue = self._calculate_revenue(user)
                primary_metric = self._get_primary_metric(test_id)
                metric_value = revenue if primary_metric == 'revenue' else 1.0
                
                self.platform.record_user_metric(
                    assignment['session_id'],
                    primary_metric,
                    metric_value
                )
            
            self.platform.complete_user_session(assignment['session_id'])
        

    def _get_primary_metric(self, test_id: str) -> str:

        try:
            test_config = self.platform.test_manager.test_configs.get(test_id)
            if test_config:
                return test_config.primary_metric
        except:
            pass
        return 'conversion' 
    
    def _calculate_conversion_probability(self, user: pd.Series) -> float:
        base_prob = 0.1
        
        if user['user_type'] == 'shopper':
            base_prob += 0.2
        if user['previous_purchases'] > 0:
            base_prob += 0.15
        if user['loyalty_score'] > 0.7:
            base_prob += 0.1
        if user['traffic_source'] == 'direct':
            base_prob += 0.05
            
        age_factor = max(0, (45 - abs(user['age'] - 35)) / 100)  
        income_factor = min(0.2, user['income'] / 500000)
        
        return min(0.8, base_prob + age_factor + income_factor)
    
    def _calculate_revenue(self, user: pd.Series) -> float:
        base_revenue = user['income'] * 0.02  
        
        if user['user_type'] == 'shopper':
            base_revenue *= 1.5
        if user['previous_purchases'] > 3:
            base_revenue *= 1.3
        if user['loyalty_score'] > 0.8:
            base_revenue *= 1.2

        noise = np.random.normal(1.0, 0.2)
        
        return max(10, base_revenue * noise)