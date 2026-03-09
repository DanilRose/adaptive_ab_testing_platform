import torch
import pandas as pd
import numpy as np
import math
import os

from typing import Optional, Dict, Any

from backend.services.safe_loader import safe_torch_load
from backend.services.traffic_generator.data_generator import RealisticDataGenerator
from backend.gan.config import GANConfig  
from backend.gan.models import GAN 

class GANService:
    def __init__(self):
        self.gan_model = None
        self.is_trained = False
        self.training_progress = 0
        self.current_status = "checkpoint_not_loaded"  # Статус: Чекпоинт не загружен
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_config_snapshot: Dict[str, Any] = {}
        self.current_config_overrides: Dict[str, Any] = {}
        self._stop_training = False
        self.loaded_checkpoint_name: Optional[str] = None  # Имя загруженного чекпоинта
    
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
        try:
            if self.gan_model is not None and not force_reinitialize and not config_overrides:
                return True

            config = GANConfig()
            if config_overrides:
                for key, value in config_overrides.items():
                   if hasattr(config, key):
                       setattr(config, key, value)

            self.gan_model = GAN(config)
            self.current_status = "checkpoint_not_loaded"  # Инициализация не меняет статус загрузки
            self.is_trained = False
            self.current_config_overrides = config_overrides or {}
            self.current_config_snapshot = self._serialize_config(config)
            return True
        except Exception as e:
            print(f"Error initializing GAN: {e}")
            self.current_status = f"error: {str(e)}"
            return False
    
    def train_gan(self, real_data: pd.DataFrame, epochs: int = 50, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            if not self.initialize_gan(config_overrides=config_overrides, force_reinitialize=True):
                return {"success": False, "error": "Failed to initialize GAN"}

            self._stop_training = False
            self.training_progress = 0
            self.current_epoch = 0
            self.total_epochs = epochs
            self.current_status = "training"  # Статус: Обучение
            
            original_train = self.gan_model.train
            original_validate = getattr(self.gan_model, '_validate_training', None)
            original_train_epoch_wgan_gp = getattr(self.gan_model, 'train_epoch_wgan_gp', None)
            original_train_epoch_standard = getattr(self.gan_model, 'train_epoch_standard', None)
            
            actual_epochs = epochs or self.gan_model.config.EPOCHS
            self.total_epochs = actual_epochs
            
            def validate_with_progress(real_data, epoch):
                if self._stop_training:
                    print(f"Обучение остановлено пользователем на эпохе {epoch}/{actual_epochs}")
                    raise KeyboardInterrupt("Training stopped by user")

                self.current_epoch = epoch
                progress = min(100, int((epoch / actual_epochs) * 100))
                self.training_progress = progress
                self.current_status = "training"  # Статус остается "training"

                print(f"Прогресс обучения: {progress}% (эпоха {epoch}/{actual_epochs})")

                if original_validate:
                    return original_validate(real_data, epoch)
                return None

            def train_epoch_wgan_gp_with_stop(dataloader, epoch):
                if self._stop_training:
                    print(f"Обучение остановлено пользователем на эпохе {epoch}/{actual_epochs}")
                    raise KeyboardInterrupt("Training stopped by user")
                self.current_epoch = epoch
                progress = min(100, int((epoch / actual_epochs) * 100))
                self.training_progress = progress
                self.current_status = "training"  # Статус остается "training"
                if original_train_epoch_wgan_gp:
                    return original_train_epoch_wgan_gp(dataloader, epoch)
                return None

            def train_epoch_standard_with_stop(dataloader, epoch):
                if self._stop_training:
                    print(f"Обучение остановлено пользователем на эпохе {epoch}/{actual_epochs}")
                    raise KeyboardInterrupt("Training stopped by user")
                self.current_epoch = epoch
                progress = min(100, int((epoch / actual_epochs) * 100))
                self.training_progress = progress
                self.current_status = "training"  # Статус остается "training"
                if original_train_epoch_standard:
                    return original_train_epoch_standard(dataloader, epoch)
                return None

            self.gan_model._validate_training = validate_with_progress
            if original_train_epoch_wgan_gp:
                self.gan_model.train_epoch_wgan_gp = train_epoch_wgan_gp_with_stop
            if original_train_epoch_standard:
                self.gan_model.train_epoch_standard = train_epoch_standard_with_stop

            result = original_train(real_data, epochs)

            if original_validate:
                self.gan_model._validate_training = original_validate
            if original_train_epoch_wgan_gp:
                self.gan_model.train_epoch_wgan_gp = original_train_epoch_wgan_gp
            if original_train_epoch_standard:
                self.gan_model.train_epoch_standard = original_train_epoch_standard
            
            if not self._stop_training:
                self.training_progress = 100
                # После успешного обучения автоматически создается чекпоинт, меняем статус
                self.current_status = "checkpoint_loaded"  # Будет обновлен после сохранения
                self.is_trained = True
            
            return {
                "success": True,
                "status": "training_completed" if not self._stop_training else "training_stopped",
                "message": f"Обучение завершено на {epochs} эпох" if not self._stop_training else f"Обучение остановлено на эпохе {self.current_epoch}/{actual_epochs}"
            }
            
        except KeyboardInterrupt:
            self.current_status = "training_paused"  # Статус: Пауза обучения
            self.training_progress = int((self.current_epoch / self.total_epochs) * 100) if self.total_epochs > 0 else 0
            return {"success": True, "status": "training_stopped", "message": f"Обучение остановлено на эпохе {self.current_epoch}/{self.total_epochs}"}
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return {"success": False, "error": str(e)}
    
    def stop_training(self) -> bool:
        if self.current_status.startswith("training"):
            self._stop_training = True
            return True
        return False
    
    def resume_training(self) -> bool:
        """
        ПРИМЕЧАНИЕ: Это демонстрационная функция. 

        """
        if self.current_status == "training_paused":
            self.current_status = "training"  
            self._stop_training = False
            return True
        return False
    
    def reset_training(self) -> bool:
        self._stop_training = False
        self.training_progress = 0
        self.current_epoch = 0
        self.total_epochs = 0
        self.is_trained = False
        self.current_status = "checkpoint_not_loaded"  
        self.loaded_checkpoint_name = None

        return True
    
    def generate_synthetic_data(self, num_samples: int = 10000, filters: Optional[Dict[str, Any]] = None, dataset_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        try:
            if self.gan_model is None or not self.is_trained:
                return None
            
            previous_status = self.current_status
            synthetic_data = self.gan_model.generate(num_samples)
            if filters:
                generator = RealisticDataGenerator()
                synthetic_data = generator.filter_dataframe(synthetic_data, filters)
            if dataset_name:
                synthetic_data['dataset_name'] = dataset_name
            self.current_status = previous_status
            
            return synthetic_data
            
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return None
    
    def get_status(self) -> Dict[str, Any]:
        status_info = {
            "status": self.current_status,
            "is_trained": self.is_trained,
            "training_progress": self.training_progress,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "has_model": self.gan_model is not None,
            "config": self.current_config_snapshot,
            "config_overrides": self.current_config_overrides,
            "loaded_checkpoint_name": self.loaded_checkpoint_name,  
        }
        
        if self.gan_model:
            g_losses_raw = self.gan_model.g_losses or []
            d_losses_raw = self.gan_model.d_losses or []
            

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
        try:
            if self.gan_model is None:
                if not self.initialize_gan():
                    return False
            
            if not os.path.exists(checkpoint_path):
                checkpoint_path = f"gan/checkpoints/{checkpoint_path}"
                if not os.path.exists(checkpoint_path):
                    return False
            
            success = self.gan_model.load_checkpoint(checkpoint_path)
            
            if success:
                self.is_trained = True
                checkpoint_name = os.path.basename(checkpoint_path)
                if checkpoint_name.endswith('.pth'):
                    checkpoint_name = checkpoint_name[:-4]
                self.loaded_checkpoint_name = checkpoint_name
                self.current_status = "checkpoint_loaded"  
                return True
            return False
            
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return False
        
gan_service = GANService()