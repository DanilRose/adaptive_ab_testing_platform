import torch

class GANConfig:
    def __init__(self):
        # Оптимальные настройки для стабильного обучения
        self.LATENT_DIM = 100
        self.BATCH_SIZE = 512
        self.EPOCHS = 300  # Увеличил для реального обучения
        self.LEARNING_RATE = 0.0002  # Стандартный LR для GAN
        
        # Используем WGAN-GP для стабильности
        self.USE_WGAN_GP = True
        self.LAMBDA_GP = 10
        self.N_CRITIC = 5
        
        # Оптимизированная архитектура
        self.GENERATOR_LAYERS = [128, 64]
        self.DISCRIMINATOR_LAYERS = [128, 64]
        
        self.DROPOUT_RATE = 0.3
        self.LEAKY_RELU_SLOPE = 0.2
        self.BETA1 = 0.5
        self.BETA2 = 0.9
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Мониторинг
        self.LOG_INTERVAL = 100
        self.CHECKPOINT_INTERVAL = 500
        self.SAMPLE_INTERVAL = 500
        self.SAMPLE_SIZE = 1000
        self.VALIDATION_INTERVAL = 200  # Новая: валидация каждые 200 эпох
        
        self.EARLY_STOPPING_THRESHOLD = 0.05
        self.EARLY_STOPPING_WINDOW = 1000