import pandas as pd
import numpy as np
from faker import Faker
import random

class RealisticDataGenerator:
    def __init__(self, seed=42):
        self.faker = Faker('ru_RU')
        self.random = random.Random(seed)
        np.random.seed(seed)
        
        self.russian_cities = [
            'Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
            'Нижний Новгород', 'Челябинск', 'Самара', 'Омск', 'Ростов-на-Дону',
            'Уфа', 'Красноярск', 'Воронеж', 'Пермь', 'Волгоград', 'Краснодар',
            'Саратов', 'Тюмень', 'Тольятти', 'Ижевск', 'Барнаул', 'Ульяновск',
            'Иркутск', 'Владивосток', 'Ярославль', 'Хабаровск', 'Махачкала',
            'Оренбург', 'Томск', 'Кемерово', 'Новокузнецк', 'Астрахань', 'Рязань',
            'Набережные Челны', 'Пенза', 'Липецк', 'Киров', 'Чебоксары', 'Тула',
            'Калининград', 'Балашиха', 'Курск', 'Севастополь', 'Сочи', 'Ставрополь',
            'Улан-Удэ', 'Тверь', 'Магнитогорск', 'Иваново', 'Брянск', 'Белгород',
            'Сургут', 'Владимир', 'Архангельск', 'Нижний Тагил', 'Чита', 'Симферополь',
            'Калуга', 'Волгодонск', 'Смоленск', 'Саранск', 'Курган', 'Волжский',
            'Орёл', 'Череповец', 'Владикавказ', 'Якутск', 'Мурманск', 'Подольск',
            'Тамбов', 'Грозный', 'Стерлитамак', 'Петрозаводск', 'Кострома', 'Нижневартовск',
            'Новороссийск', 'Йошкар-Ола', 'Химки', 'Таганрог', 'Сыктывкар', 'Нальчик',
            'Шахты', 'Дзержинск', 'Орск', 'Братск', 'Ангарск', 'Энгельс', 'Благовещенск',
            'Великий Новгород', 'Королёв', 'Псков', 'Бийск', 'Прокопьевск', 'Рыбинск',
            'Балаково', 'Армавир', 'Южно-Сахалинск', 'Северодвинск', 'Абакан', 'Петропавловск-Камчатский',
            'Норильск', 'Сызрань', 'Волжск', 'Каменск-Уральский', 'Новочеркасск', 'Златоуст',
            'Электросталь', 'Салават', 'Миасс', 'Находка', 'Керчь', 'Копейск', 'Хасавюрт',
            'Уссурийск', 'Димитровград', 'Артём', 'Новоуральск', 'Серпухов', 'Бердск',
            'Новомосковск', 'Первоуральск', 'Нефтеюганск', 'Кисловодск', 'Обнинск', 'Красногорск',
            'Муром', 'Батайск', 'Елец', 'Пятигорск', 'Ковров', 'Реутов', 'Северск', 'Назрань',
            'Новошахтинск', 'Железнодорожный', 'Каспийск', 'Дербент', 'Октябрьский', 'Новотроицк',
            'Нефтекамск', 'Щёлково', 'Кызыл', 'Сергиев Посад', 'Ачинск', 'Арзамас', 'Ессентуки',
            'Новый Уренгой', 'Ленинск-Кузнецкий', 'Жуковский', 'Междуреченск', 'Саров', 'Элиста',
            'Зеленогорск', 'Солнечногорск', 'Глазов', 'Великие Луки', 'Канск', 'Киселёвск',
            'Мичуринск', 'Губкин', 'Ухта', 'Бугульма', 'Елабуга', 'Ступино', 'Азов', 'Бор',
            'Чайковский', 'Лобня', 'Минеральные Воды', 'Анжеро-Судженск', 'Биробиджан', 'Лесосибирск',
            'Кушва', 'Черкесск', 'Нерюнгри', 'Шадринск', 'Тобольск', 'Ялта', 'Выборг', 'Белово',
            'Курчатов', 'Лысьва', 'Чапаевск', 'Салехард', 'Горно-Алтайск', 'Мелеуз', 'Когалым',
            'Краснокаменск', 'Тихвин', 'Александров', 'Сосновый Бор', 'Гусь-Хрустальный', 'Воткинск',
            'Минусинск', 'Жигулёвск', 'Кинешма', 'Реж', 'Верхняя Пышма', 'Лениногорск', 'Славгород',
            'Краснокамск', 'Саянск', 'Рузаевка', 'Трёхгорный', 'Бузулук', 'Асбест', 'Гатчина',
            'Воркута', 'Кстово', 'Ишимбай', 'Шуя', 'Волхов', 'Всеволожск', 'Кунгур', 'Борисоглебск',
            'Белорецк', 'Зеленодольск', 'Лесной', 'Черногорск', 'Павловский Посад', 'Красноперекопск',
            'Кизляр', 'Раменское', 'Донской', 'Сертолово', 'Сосновоборск', 'Моршанск', 'Климовск',
            'Люберцы', 'Балей', 'Краснотурьинск', 'Кропоткин', 'Феодосия', 'Белореченск', 'Ивантеевка',
            'Котовск', 'Видное', 'Георгиевск', 'Лангепас', 'Снежинск', 'Урус-Мартан', 'Будённовск',
            'Алушта', 'Советск', 'Наро-Фоминск', 'Полевской', 'Лыткарино', 'Россошь', 'Тихорецк',
            'Анапа', 'Алексин', 'Черняховск', 'Костомукша', 'Железногорск', 'Курганинск', 'Волгодонск',
            'Усть-Илимск', 'Лесозаводск', 'Кандалакша', 'Свободный', 'Златоуст', 'Кириши', 'Краснознаменск',
            'Балашов', 'Тейково', 'Шлиссельбург', 'Бахчисарай', 'Гуково', 'Петухово', 'Славянск-на-Кубани',
            'Красный Сулин', 'Мценск', 'Фролово', 'Лабинск', 'Тара', 'Светлогорск', 'Дальнегорск',
            'Карасук', 'Кувандык', 'Майкоп', 'Белогорск', 'Кизел', 'Лабытнанги', 'Шатура', 'Александровск',
            'Соль-Илецк', 'Тутаев', 'Красный Луч', 'Кирсанов', 'Плавск', 'Советская Гавань', 'Бирск',
            'Кулебаки', 'Верхняя Салда', 'Инза', 'Краснослободск', 'Моздок', 'Новоалтайск', 'Лиски',
            'Бологое', 'Дно', 'Галич', 'Кимры', 'Конаково', 'Нея', 'Остров', 'Пестово', 'Солигалич',
            'Старая Русса', 'Торжок', 'Удомля', 'Холм', 'Чудово', 'Шимск', 'Бокситогорск', 'Волосово',
            'Волхов', 'Всеволожск', 'Выборг', 'Гатчина', 'Кингисепп', 'Кириши', 'Кировск', 'Лодейное Поле',
            'Ломоносов', 'Луга', 'Подпорожье', 'Приозерск', 'Сланцы', 'Тихвин', 'Тосно'
        ]
        
        self.device_os_browser_mapping = {
            'Mobile': {
                'iOS': ['Safari Mobile', 'Chrome Mobile'],
                'Android': ['Chrome Mobile', 'Firefox Mobile', 'Samsung Internet']
            },
            'Desktop': {
                'Windows': ['Chrome', 'Firefox', 'Edge'],
                'macOS': ['Safari', 'Chrome', 'Firefox'],
                'Linux': ['Firefox', 'Chrome']
            },
            'Tablet': {
                'iOS': ['Safari Mobile', 'Chrome Mobile'],
                'Android': ['Chrome Mobile', 'Firefox Mobile']
            }
        }
        
        self.os_weights = {
            'Mobile': {'iOS': 0.4, 'Android': 0.6},
            'Desktop': {'Windows': 0.7, 'macOS': 0.25, 'Linux': 0.05},
            'Tablet': {'iOS': 0.3, 'Android': 0.7}
        }

    def _get_os_and_browser(self, device):
        os_weights = self.os_weights[device]
        os = self.random.choices(list(os_weights.keys()), weights=list(os_weights.values()))[0]
        available_browsers = self.device_os_browser_mapping[device][os]
        browser = self.random.choice(available_browsers)
        return os, browser

    def generate_user(self, user_id):
        if self.random.random() < 0.6:
            age = int(np.random.normal(28, 5))
        else:
            age = int(np.random.normal(45, 8))
        age = max(18, min(70, age))
        
        gender_bias = 0.5 + (age - 30) * 0.008
        gender = 'Male' if self.random.random() < gender_bias else 'Female'
        
        city = self.random.choice(self.russian_cities)
        
        age_group_probs = {
            '18-25': {'Mobile': 0.8, 'Desktop': 0.15, 'Tablet': 0.05},
            '26-35': {'Mobile': 0.6, 'Desktop': 0.3, 'Tablet': 0.1},
            '36-50': {'Mobile': 0.4, 'Desktop': 0.5, 'Tablet': 0.1},
            '50+': {'Mobile': 0.2, 'Desktop': 0.7, 'Tablet': 0.1}
        }
        
        if age <= 25: age_group = '18-25'
        elif age <= 35: age_group = '26-35' 
        elif age <= 50: age_group = '36-50'
        else: age_group = '50+'
        
        device_probs = age_group_probs[age_group]
        device = self.random.choices(list(device_probs.keys()), weights=list(device_probs.values()))[0]
        
        os, browser = self._get_os_and_browser(device)
        
        if age < 30:
            traffic_source = self.random.choices(['social', 'organic', 'direct', 'referral'], weights=[0.4, 0.3, 0.2, 0.1])[0]
        else:
            traffic_source = self.random.choices(['direct', 'organic', 'email', 'social'], weights=[0.4, 0.3, 0.2, 0.1])[0]
        
        income_base = 35000 if age_group == '18-25' else 65000 if age_group == '26-35' else 85000 if age_group == '36-50' else 70000
        income = int(np.random.normal(income_base, 15000))
        income = max(20000, income)
        
        purchase_freq_base = 0.3 if age_group == '18-25' else 0.6 if age_group == '26-35' else 0.8 if age_group == '36-50' else 0.5
        previous_purchases = max(0, int(np.random.poisson(purchase_freq_base * 5)))
        
        total_spent = previous_purchases * np.random.normal(1500, 500)
        avg_order_value = total_spent / max(1, previous_purchases)
        
        if device == 'Mobile':
            session_duration = max(30, int(np.random.normal(180, 60)))
            pages_per_session = max(3, int(np.random.poisson(8)))
            bounce_rate = np.random.beta(2, 5)
        else:
            session_duration = max(60, int(np.random.normal(300, 100)))
            pages_per_session = max(5, int(np.random.poisson(12)))
            bounce_rate = np.random.beta(3, 4)
        
        days_since_signup = np.random.poisson(180)
        loyalty_score = min(1.0, days_since_signup / 365 * 0.7)
        
        if loyalty_score > 0.7:
            visits_per_week = np.random.poisson(3)
        else:
            visits_per_week = np.random.poisson(1.5)
        
        email_subscribed = self.random.random() < 0.6
        push_enabled = self.random.random() < 0.4
        has_newsletter = self.random.random() < 0.3
        
        return {
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'city': city,
            'income': income,
            'device': device,
            'os': os,
            'browser': browser,
            'screen_resolution': f"{self.random.randint(1200, 3840)}x{self.random.randint(800, 2160)}",
            'previous_purchases': previous_purchases,
            'total_spent': total_spent,
            'avg_order_value': avg_order_value,
            'session_duration': session_duration,
            'pages_per_session': pages_per_session,
            'bounce_rate': bounce_rate,
            'visits_per_week': visits_per_week,
            'traffic_source': traffic_source,
            'days_since_signup': days_since_signup,
            'loyalty_score': loyalty_score,
            'email_subscribed': email_subscribed,
            'push_enabled': push_enabled,
            'has_newsletter': has_newsletter,
            'time_on_site': np.random.normal(300, 100),
            'pages_visited': np.random.poisson(15),
            'cart_additions': np.random.poisson(3),
            'wishlist_items': np.random.poisson(2),
            'product_views': np.random.poisson(25),
            'search_queries': np.random.poisson(5),
            'filter_usage': np.random.poisson(2),
            'sort_usage': np.random.beta(2, 5),
            'reviews_written': np.random.poisson(1),
            'ratings_given': np.random.poisson(2),
            'social_shares': np.random.poisson(0.5),
            'coupons_used': np.random.poisson(1),
            'discounts_used': np.random.poisson(2),
            'returns_count': np.random.poisson(0.3),
            'complaints_count': np.random.poisson(0.1),
            'support_contacts': np.random.poisson(0.5),
            'app_downloads': 1 if device == 'Mobile' else 0,
            'push_notification_clicks': np.random.poisson(3),
            'email_opens': np.random.poisson(5),
            'email_clicks': np.random.poisson(2),
            'sms_received': np.random.poisson(1),
            'sms_clicks': np.random.poisson(0.5),
            'affiliate_clicks': np.random.poisson(1),
            'referral_signups': np.random.poisson(0.2),
            'loyalty_points': np.random.poisson(50),
            'tier_level': self.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum']),
            'birthday_month': self.random.randint(1, 12),
            'anniversary_date': self.faker.date_between(start_date='-5y', end_date='today'),
            'last_purchase_date': self.faker.date_between(start_date='-90d', end_date='today'),
            'first_purchase_date': self.faker.date_between(start_date='-2y', end_date='today'),
            'preferred_category': self.random.choice(['Electronics', 'Fashion', 'Home', 'Beauty', 'Sports', 'Books']),
            'preferred_brand': self.random.choice(['Apple', 'Samsung', 'Nike', 'Adidas', 'Sony', 'LG']),
            'preferred_price_range': self.random.choice(['Budget', 'Mid-range', 'Premium']),
            'preferred_payment_method': self.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay']),
            'preferred_shipping_method': self.random.choice(['Standard', 'Express', 'Next Day']),
            'preferred_communication_channel': self.random.choice(['Email', 'SMS', 'Push', 'None']),
            'preferred_discount_type': self.random.choice(['Percentage', 'Fixed', 'Free Shipping']),
            'preferred_product_type': self.random.choice(['New', 'Sale', 'Bestseller']),
            'preferred_delivery_time': self.random.choice(['Morning', 'Afternoon', 'Evening']),
            'device_age': np.random.poisson(12),
            'browser_version': f"{self.random.randint(10, 15)}.{self.random.randint(0, 9)}",
            'app_version': f"{self.random.randint(1, 5)}.{self.random.randint(0, 9)}",
            'connection_type': self.random.choice(['WiFi', '4G', '5G', '3G']),
            'location_accuracy': np.random.normal(50, 20),
            'timezone': self.random.choice(['UTC+3', 'UTC+4', 'UTC+5', 'UTC+6', 'UTC+7']),
            'language': self.random.choice(['ru', 'en', 'uk', 'kz']),
            'cookie_enabled': self.random.random() < 0.95,
            'javascript_enabled': self.random.random() < 0.98,
            'flash_enabled': self.random.random() < 0.1,
            'images_enabled': self.random.random() < 0.99,
            'css_enabled': self.random.random() < 0.99,
            'screen_color_depth': self.random.choice([16, 24, 32]),
            'screen_pixel_ratio': round(np.random.uniform(1.0, 3.0), 2),
            'device_memory': self.random.choice([2, 4, 8, 16]),
            'hardware_concurrency': self.random.choice([2, 4, 6, 8]),
            'max_touch_points': self.random.randint(0, 10),
            'referrer_domain': self.random.choice(['google.com', 'yandex.ru', 'mail.ru', 'direct', 'social']),
            'utm_source': self.random.choice(['google', 'yandex', 'email', 'social']),
            'utm_medium': self.random.choice(['cpc', 'organic', 'email', 'social']),
            'utm_campaign': self.random.choice(['spring_sale', 'winter_sale', 'new_collection']),
            'utm_term': self.random.choice(['buy+shoes', 'electronics+online', 'fashion+sale']),
            'utm_content': self.random.choice(['banner', 'text', 'video']),
            'gclid': self.faker.uuid4() if self.random.random() < 0.3 else '',
            'fbclid': self.faker.uuid4() if self.random.random() < 0.2 else '',
            'msclkid': self.faker.uuid4() if self.random.random() < 0.1 else '',
            'session_count': np.random.poisson(15),
            'session_duration_avg': np.random.normal(240, 80),
            'session_depth_avg': np.random.poisson(12),
            'conversion_rate': np.random.beta(2, 8),
            'cart_abandonment_rate': np.random.beta(3, 2),
            'product_return_rate': np.random.beta(1, 9),
            'customer_satisfaction_score': np.random.normal(4.2, 0.5),
            'net_promoter_score': np.random.randint(0, 10),
            'customer_effort_score': np.random.randint(1, 7),
            'churn_probability': np.random.beta(2, 8),
            'lifetime_value': np.random.normal(5000, 2000),
            'acquisition_cost': np.random.normal(50, 20),
            'margin_contribution': np.random.normal(0.3, 0.1),
            'predicted_ltv': np.random.normal(6000, 2500),
            'segmentation_tier': self.random.choice(['High Value', 'Medium Value', 'Low Value']),
            'rfm_score': self.random.randint(1, 555),
            'clustering_group': self.random.randint(1, 10),
            'personalization_score': np.random.beta(5, 2),
            'engagement_score': np.random.beta(4, 3),
            'retention_score': np.random.beta(3, 4),
            'vip_status': self.random.random() < 0.1,
            'beta_tester': self.random.random() < 0.05,
            'early_adopter': self.random.random() < 0.2
        }

    def generate_dataset(self, n_samples=10000):
        users = []
        for i in range(n_samples):
            if i % 10000 == 0:
                print(f"Генерация данных: {i}/{n_samples}")
            user = self.generate_user(i)
            users.append(user)
        
        df = pd.DataFrame(users)
        print(f"Сгенерировано {len(df)} пользователей с {len(df.columns)} признаками")
        return df