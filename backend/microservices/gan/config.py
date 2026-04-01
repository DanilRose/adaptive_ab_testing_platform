import torch

class GANConfig:
    def __init__(self):
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # настройки обучения
        self.LATENT_DIM = 256 # Размер шума. Чем ниже параметр - тем проще модель
        self.BATCH_SIZE = 1024 # Размер батча. Влияет на кол-во памяти при обучении. Чем выше — быстрее и больше памяти
        self.EPOCHS = 50  # Кол-во эпох. Влияет на качество обучения
        self.LEARNING_RATE = 1e-4 # шаг обучения
        self.BETA1 = 0.0 # настройка для WGAN_GP. ВСЕГДА ДОЛЖНА БЫТЬ НОЛЬ ИНАЧЕ ВЗРЫВ
        self.BETA2 = 0.9 # Адаптация, чем ниже - тем хуже
        self.USE_WGAN_GP = True # лучше применять WGAN_GP режим, иначе данные будут слишком синтетические
        self.LAMBDA_GP = 10 # контроль за градиентом
        self.N_CRITIC = 5 # чем выше, тем дискриминатор пытается создать реалистичные данные. Нельзя слишком высокие значения
        self.GENERATOR_LAYERS = [512, 512, 256, 256, 128] # слои дискриминатора
        self.DISCRIMINATOR_LAYERS = [512, 512, 256, 256, 128] # слои генератора
        self.DROPOUT_RATE = 0.1 # влияет на простоту модели. слишком высокие значения - меньше переобучения
        self.LEAKY_RELU_SLOPE = 0.2 # наклон leakyrelu
        
        # просмотр обучения
        self.LOG_INTERVAL = 10 # промежуток эпох печати логов
        self.CHECKPOINT_INTERVAL = 100 # промежуток эпох сохранения чекпоинта
        self.SAMPLE_INTERVAL = 10 # генерация сэмплов
        self.SAMPLE_SIZE = 1000 # сколько сэмплов
        self.VALIDATION_INTERVAL = 5  #FID проверка каждые n эпох
        
        # ранняя остановка
        self.EARLY_STOPPING_THRESHOLD = 0.05
        self.EARLY_STOPPING_WINDOW = 1000