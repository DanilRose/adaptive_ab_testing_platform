# backend/services/gan_integration.py
import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
from backend.services.safe_loader import safe_torch_load
from backend.services.traffic_generator.data_generator import RealisticDataGenerator

class GANService:
    def __init__(self):
        self.gan_model = None
        self.is_trained = False
        self.training_progress = 0
        self.current_status = "not_initialized"
        self.current_epoch = 0
        self.total_epochs = 0
        self.available_checkpoints = []
        self.current_config_snapshot: Dict[str, Any] = {}
        self.current_config_overrides: Dict[str, Any] = {}
        self._stop_training = False

    def set_available_checkpoints(self, checkpoints: list[dict[str, Any]]):

        self.available_checkpoints = checkpoints or []
    
    def _is_gan_checkpoint(self, filepath: str) -> bool:
        """Проверка чекпоинта через безопасную загрузку"""
        try:
            checkpoint = safe_torch_load(filepath, map_location='cpu')
            has_generator = 'generator_state_dict' in checkpoint or 'generator' in checkpoint
            has_discriminator = 'discriminator_state_dict' in checkpoint or 'discriminator' in checkpoint
            return has_generator or has_discriminator
        except:
            return False
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        if config is None:
            return snapshot
        
        # Если config - это словарь, используем его напрямую
        if isinstance(config, dict):
            for key, value in config.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, torch.device):
                    snapshot[key] = str(value)
                elif isinstance(value, (list, tuple)):
                    snapshot[key] = list(value)
                elif isinstance(value, (int, float, str, bool, type(None))):
                    snapshot[key] = value
                else:
                    snapshot[key] = str(value)
            return snapshot
        
        # Если config - это объект с __dict__
        for key, value in config.__dict__.items():
            if key.startswith("_"):
                continue
            if isinstance(value, torch.device):
                snapshot[key] = str(value)
            elif isinstance(value, (list, tuple)):
                snapshot[key] = list(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                snapshot[key] = value
            else:
                snapshot[key] = str(value)
        return snapshot

    def initialize_gan(self, config_overrides: Optional[Dict[str, Any]] = None, force_reinitialize: bool = False):
        """Инициализация GAN модели"""
        try:
            from backend.gan.config import GANConfig  # ← ПРОВЕРЬ ПУТЬ!
            from backend.gan.models import GAN        # ← ПРОВЕРЬ ПУТЬ!

            if self.gan_model is not None and not force_reinitialize and not config_overrides:
                return True

            config = GANConfig()
            if config_overrides:
                for key, value in config_overrides.items():
                   if hasattr(config, key):
                       setattr(config, key, value)

            self.gan_model = GAN(config)      # ← ТЕПЕРЬ ПРАВИЛЬНЫЙ КЛАСС
            self.current_status = "initialized"
            self.is_trained = False
            self.current_config_overrides = config_overrides or {}
            # Сериализуем конфигурацию ПОСЛЕ применения overrides
            self.current_config_snapshot = self._serialize_config(config)
            return True
        except Exception as e:
            print(f"Error initializing GAN: {e}")
            self.current_status = f"error: {str(e)}"
            return False
    
    def train_gan(self, real_data: pd.DataFrame, epochs: int = 50, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Обучение GAN модели с обновлением прогресса (синхронно)"""
        try:
            if not self.initialize_gan(config_overrides=config_overrides, force_reinitialize=True):
                return {"success": False, "error": "Failed to initialize GAN"}

            self._stop_training = False
            self.current_status = "training_0%"
            self.training_progress = 0
            self.current_epoch = 0
            self.total_epochs = epochs
            
            # Сохраняем оригинальные методы
            original_train = self.gan_model.train
            original_validate = getattr(self.gan_model, '_validate_training', None)
            
            actual_epochs = epochs or self.gan_model.config.EPOCHS
            self.total_epochs = actual_epochs
            
            # Переопределяем валидацию для отслеживания прогресса
            def validate_with_progress(real_data, epoch):
                # Проверка флага остановки
                if self._stop_training:
                    print(f"Обучение остановлено пользователем на эпохе {epoch}/{actual_epochs}")
                    raise KeyboardInterrupt("Training stopped by user")
                
                self.current_epoch = epoch
                progress = min(100, int((epoch / actual_epochs) * 100))
                self.training_progress = progress
                self.current_status = f"training_{progress}% (эпоха {epoch}/{actual_epochs})"
                
                print(f"Прогресс обучения: {progress}% (эпоха {epoch}/{actual_epochs})")
                
                if original_validate:
                    return original_validate(real_data, epoch)
                return None
            
            self.gan_model._validate_training = validate_with_progress
            
            # Запускаем обучение СИНХРОННО (вызывающий код сам решает, запускать в потоке или нет)
            result = original_train(real_data, epochs)
            
            # Завершение обучения
            if not self._stop_training:
                self.training_progress = 100
                self.current_status = "trained"
                self.is_trained = True
            
            return {
                "success": True,
                "status": "training_completed" if not self._stop_training else "training_stopped",
                "message": f"Обучение завершено на {epochs} эпох" if not self._stop_training else f"Обучение остановлено на эпохе {self.current_epoch}/{actual_epochs}"
            }
            
        except KeyboardInterrupt:
            self.current_status = "stopped"
            self.training_progress = int((self.current_epoch / self.total_epochs) * 100) if self.total_epochs > 0 else 0
            return {"success": True, "status": "training_stopped", "message": f"Обучение остановлено на эпохе {self.current_epoch}/{self.total_epochs}"}
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return {"success": False, "error": str(e)}
    
    def stop_training(self) -> bool:
        """Остановка обучения GAN"""
        if self.current_status.startswith("training"):
            self._stop_training = True
            return True
        return False
    
    def generate_synthetic_data(self, num_samples: int = 10000, filters: Optional[Dict[str, Any]] = None, dataset_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Генерация синтетических данных"""
        try:
            if self.gan_model is None or not self.is_trained:
                return None
            
            self.current_status = "generating"
            synthetic_data = self.gan_model.generate(num_samples)
            if filters:
                generator = RealisticDataGenerator()
                synthetic_data = generator.filter_dataframe(synthetic_data, filters)
            if dataset_name:
                synthetic_data['dataset_name'] = dataset_name
            self.current_status = "ready"
            
            return synthetic_data
            
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса GAN сервиса"""
        status_info = {
            "status": self.current_status,
            "is_trained": self.is_trained,
            "training_progress": self.training_progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "has_model": self.gan_model is not None,
            "available_checkpoints": len(self.available_checkpoints),
            "checkpoints": self.available_checkpoints[:10],  # Только последние 10
            "config": self.current_config_snapshot,
            "config_overrides": self.current_config_overrides,
        }
        
        if self.gan_model:
            g_losses_raw = self.gan_model.g_losses or []
            d_losses_raw = self.gan_model.d_losses or []
            
            # Фильтруем None и non-finite значения из потерь
            import math
            g_losses = [v for v in g_losses_raw if v is not None and not math.isnan(v) and not math.isinf(v)]
            d_losses = [v for v in d_losses_raw if v is not None and not math.isnan(v) and not math.isinf(v)]
            
            status_info["loss_history"] = {
                "g_losses": g_losses[-10:],
                "d_losses": d_losses[-10:],
                "total_epochs": len(g_losses),
                "latest_g_loss": g_losses[-1] if g_losses else None,
                "latest_d_loss": d_losses[-1] if d_losses else None
            }
            
            if hasattr(self.gan_model, 'wasserstein_distances') and self.gan_model.wasserstein_distances:
                wasserstein_raw = self.gan_model.wasserstein_distances
                wasserstein = [v for v in wasserstein_raw if v is not None and not math.isnan(v) and not math.isinf(v)]
                status_info["loss_history"]["wasserstein"] = wasserstein[-10:]
                status_info["loss_history"]["latest_wasserstein"] = wasserstein[-1] if wasserstein else None
        
        return status_info
    
    def load_pretrained_model(self, checkpoint_path: str) -> bool:
        """Загрузка предобученной модели"""
        try:
            if self.gan_model is None:
                if not self.initialize_gan():
                    return False
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = f"gan/checkpoints/{checkpoint_path}"
                if not os.path.exists(checkpoint_path):
                    return False
            
            # Используем встроенный метод GAN класса для загрузки
            success = self.gan_model.load_checkpoint(checkpoint_path)
            
            if success:
                self.is_trained = True
                self.current_status = f"loaded: {os.path.basename(checkpoint_path)}"
                return True
            return False
            
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return False
# Глобальный экземпляр сервиса
gan_service = GANService()