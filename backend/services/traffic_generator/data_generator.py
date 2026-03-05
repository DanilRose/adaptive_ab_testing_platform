import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
from typing import Any, Dict, List, Optional


class RealisticDataGenerator:
    def __init__(self, seed=42):
        self.faker = Faker('ru_RU')
        self.random = random.Random(seed)
        np.random.seed(seed)

        self.russian_cities = [
            'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
            'Нижний Новгород', 'Челябинск', 'Самара', 'Омск', 'Ростов-на-Дону'
        ]

        self.device_profiles = {
            'Mobile': {
                'os': ['iOS', 'Android'], 
                'browsers': {
                    'iOS': ['Safari Mobile', 'Chrome Mobile'],
                    'Android': ['Chrome Mobile', 'Firefox Mobile', 'Samsung Internet']
                }
            },
            'Desktop': {
                'os': ['Windows', 'macOS'], 
                'browsers': {
                    'Windows': ['Chrome', 'Firefox', 'Edge'],
                    'macOS': ['Safari', 'Chrome', 'Firefox']
                }
            },
            'Tablet': {
                'os': ['iOS', 'Android'],
                'browsers': {
                    'iOS': ['Safari Mobile', 'Chrome Mobile'],
                    'Android': ['Chrome Mobile', 'Firefox Mobile', 'Samsung Internet']
                }
            }
        }

    def generate_user(self, user_id):
        age_group = np.random.choice(['student', 'young_pro', 'family', 'senior'], 
                                    p=[0.25, 0.35, 0.3, 0.1])

        if age_group == 'student':
            age = np.random.normal(21, 2)  
        elif age_group == 'young_pro':  
            age = np.random.normal(32, 4)  
        elif age_group == 'family':
            age = np.random.normal(45, 5)  
        else:  
            age = np.random.normal(58, 4) 

        age = max(18, min(70, int(age)))
        gender = 'Male' if np.random.random() < 0.5 else 'Female'
        city = self.random.choice(self.russian_cities)

        hour_of_day = np.random.randint(0, 24)
        is_weekend = np.random.random() < 0.3  

        if 6 <= hour_of_day <= 12:
            session_quality = np.random.normal(1.2, 0.2)  
        elif 0 <= hour_of_day <= 6:
            session_quality = np.random.normal(0.8, 0.3)  
        else:
            session_quality = np.random.normal(1.0, 0.1) 

        user_type = np.random.choice(['browser', 'shopper', 'researcher', 'returning'], 
                                    p=[0.4, 0.2, 0.2, 0.2])

        base_income = 30000 + (age - 18) * 800  
        city_multiplier = 1.5 if city in ['Москва', 'Санкт-Петербург'] else 1.0

        if user_type == 'shopper':
            income_multiplier = 1.3  
        elif user_type == 'returning':
            income_multiplier = 1.2  
        else:
            income_multiplier = 1.0

        income = max(20000, int(np.random.normal(base_income * city_multiplier * income_multiplier, 15000)))

        if age < 30:
            device = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.7, 0.2, 0.1])
        else:
            device = np.random.choice(['Mobile', 'Desktop', 'Tablet'], p=[0.4, 0.5, 0.1])
        
        profile = self.device_profiles[device]
        os = self.random.choice(profile['os'])

        browser = self.random.choice(profile['browsers'][os])
        
        if device == 'Mobile':
            base_session_duration = max(30, int(np.random.normal(180, 50)))
            base_pages_per_session = max(3, int(np.random.poisson(8)))
        elif device == 'Tablet':
            base_session_duration = max(45, int(np.random.normal(240, 60)))
            base_pages_per_session = max(4, int(np.random.poisson(10)))
        else: 
            base_session_duration = max(60, int(np.random.normal(300, 80)))
            base_pages_per_session = max(5, int(np.random.poisson(12)))
        

        if user_type == 'browser':
            pages_multiplier = 0.8
            duration_multiplier = 0.9
            purchase_probability = 0.05
        elif user_type == 'shopper':
            pages_multiplier = 1.3
            duration_multiplier = 1.2
            purchase_probability = 0.3
        elif user_type == 'researcher':
            pages_multiplier = 1.5
            duration_multiplier = 1.3
            purchase_probability = 0.1
        else:  
            pages_multiplier = 1.1
            duration_multiplier = 1.0
            purchase_probability = 0.25

        session_duration = int(base_session_duration * duration_multiplier * session_quality)
        pages_per_session = int(base_pages_per_session * pages_multiplier * session_quality)
        
        purchase_probability += (age - 18) * 0.002 + (income / 100000) * 0.05
        previous_purchases = np.random.poisson(purchase_probability * 10)
        
        avg_spend_per_purchase = np.random.normal(income * 0.02, income * 0.005)
        total_spent = previous_purchases * avg_spend_per_purchase
        total_spent = np.clip(total_spent, 0, 300000)

        days_since_signup = np.random.exponential(365)
        loyalty_score = min(1.0, np.sqrt(days_since_signup / 365))

        base_visits = 1 + loyalty_score * 3
        if user_type == 'returning':
            visits_multiplier = 1.5
        elif user_type == 'researcher':
            visits_multiplier = 1.3
        else:
            visits_multiplier = 1.0
            
        visits_per_week = np.random.poisson(base_visits * visits_multiplier)

        if user_type == 'shopper':
            traffic_source = np.random.choice(['social', 'direct', 'email'], p=[0.4, 0.4, 0.2])
        elif user_type == 'researcher':
            traffic_source = np.random.choice(['organic', 'direct', 'social'], p=[0.5, 0.3, 0.2])
        elif age < 25:
            traffic_source = np.random.choice(['social', 'organic', 'direct'], p=[0.5, 0.3, 0.2])
        elif age < 35:
            traffic_source = np.random.choice(['social', 'organic', 'direct', 'email'], p=[0.4, 0.3, 0.2, 0.1])
        else:
            traffic_source = np.random.choice(['direct', 'organic', 'email'], p=[0.5, 0.3, 0.2])
        
        email_prob = 0.7 if age > 25 else 0.4
        if user_type == 'returning':
            email_prob += 0.2
        email_subscribed = np.random.random() < email_prob
        
        push_enabled = device in ['Mobile', 'Tablet'] and np.random.random() < 0.7
        
        return {
            'age': age,
            'gender': gender,
            'city': city,
            'income': income,
            'device': device,
            'os': os,
            'browser': browser,
            'session_duration': session_duration,
            'pages_per_session': pages_per_session,
            'previous_purchases': previous_purchases,
            'total_spent': total_spent,
            'loyalty_score': loyalty_score,
            'visits_per_week': visits_per_week,
            'traffic_source': traffic_source,
            'email_subscribed': email_subscribed,
            'push_enabled': push_enabled,
            'user_type': user_type,
            'hour_of_day': hour_of_day,
            'is_weekend': is_weekend,
            'session_quality': round(session_quality, 2),
        }

    def _ensure_list(self, value: Any) -> Optional[List[Any]]:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return [value]

    def _matches_filters(self, user: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True

        field_map = {
            'cities': 'city',
            'devices': 'device',
            'os': 'os',
            'browsers': 'browser',
            'user_types': 'user_type',
            'traffic_sources': 'traffic_source',
            'genders': 'gender',
        }

        for request_key, user_key in field_map.items():
            allowed_values = self._ensure_list(filters.get(request_key))
            if allowed_values and user.get(user_key) not in allowed_values:
                return False

        boolean_fields = ['email_subscribed', 'push_enabled', 'is_weekend']
        for field in boolean_fields:
            if field in filters and filters[field] is not None:
                if bool(user.get(field)) != bool(filters[field]):
                    return False

        numeric_ranges = filters.get('numeric_ranges') or {}
        for key, bounds in numeric_ranges.items():
            value = user.get(key)
            if value is None:
                continue
            min_value = bounds.get('min')
            max_value = bounds.get('max')
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False

        return True

    def filter_dataframe(self, df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not filters:
            return df
        mask = df.apply(lambda row: self._matches_filters(row.to_dict(), filters), axis=1)
        return df[mask]

    def generate_dataset(self, n_samples=50000, filters: Optional[Dict[str, Any]] = None):
        users = []
        attempts = 0
        max_attempts = n_samples * (20 if filters else 1)

        while len(users) < n_samples and attempts < max_attempts:
            if attempts % 10000 == 0:
                print(f"Генерация данных: {len(users)}/{n_samples} (попыток: {attempts})")
            user = self.generate_user(attempts)
            attempts += 1

            if filters and not self._matches_filters(user, filters):
                continue

            users.append(user)

        if len(users) < n_samples:
            raise ValueError("Не удалось сгенерировать выборку с заданными фильтрами. Смягчите параметры.")

        df = pd.DataFrame(users)
        df.insert(0, 'user_id', range(len(df)))
        return df

    def get_filter_options(self) -> dict:
        browser_values = set()
        os_values = set()
        for profile in self.device_profiles.values():
            for os_value, browsers in profile['browsers'].items():
                os_values.add(os_value)
                browser_values.update(browsers)

        return {
            'cities': list(self.russian_cities),
            'devices': list(self.device_profiles.keys()),
            'os': sorted(list(os_values)),
            'browsers': sorted(list(browser_values)),
            'user_types': ['browser', 'shopper', 'researcher', 'returning'],
            'traffic_sources': ['social', 'direct', 'email', 'organic'],
            'genders': ['Male', 'Female'],
            'boolean_fields': ['email_subscribed', 'push_enabled', 'is_weekend'],
            'numeric_ranges': {
                'age': {'min': 18, 'max': 70},
                'income': {'min': 20000, 'max': 200000},
                'session_duration': {'min': 30, 'max': 600},
                'pages_per_session': {'min': 1, 'max': 25},
                'previous_purchases': {'min': 0, 'max': 40},
                'total_spent': {'min': 0, 'max': 300000},
                'visits_per_week': {'min': 0, 'max': 25},
            }
        }

if __name__ == "__main__":
    generator = RealisticDataGenerator()
    test_data = generator.generate_dataset(1000)