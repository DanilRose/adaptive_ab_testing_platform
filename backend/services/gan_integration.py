# backend/services/gan_integration.py
import torch
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
import glob
from datetime import datetime
from backend.services.safe_loader import safe_torch_load

class GANService:
    def __init__(self):
        self.gan_model = None
        self.is_trained = False
        self.training_progress = 0
        self.current_status = "not_initialized"
        self.current_epoch = 0
        self.total_epochs = 0
        self.available_checkpoints = []
        self._load_available_checkpoints()
    
    def _load_available_checkpoints(self):
        """Загрузка списка доступных чекпоинтов из всех возможных папок"""
        search_paths = [
            "gan/checkpoints/*.pth",
            "backend/gan/checkpoints/*.pth", 
            "gan_checkpoint_*.pth",
            "*.pth"
        ]
        
        self.available_checkpoints = []
        for path in search_paths:
            checkpoint_files = glob.glob(path)
            for file in checkpoint_files:
                try:
                    file_info = {
                        'filename': file,
                        'size': os.path.getsize(file),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file)),
                        'path': os.path.abspath(file)
                    }
                    # Проверяем, что это действительно чекпоинт GAN
                    if self._is_gan_checkpoint(file):
                        self.available_checkpoints.append(file_info)
                except:
                    continue
        
        # Сортируем по дате изменения (новые сверху)
        self.available_checkpoints.sort(key=lambda x: x['modified'], reverse=True)
    
    def _is_gan_checkpoint(self, filepath: str) -> bool:
        """Проверка чекпоинта через безопасную загрузку"""
        try:
            checkpoint = safe_torch_load(filepath, map_location='cpu')
            has_generator = 'generator_state_dict' in checkpoint or 'generator' in checkpoint
            has_discriminator = 'discriminator_state_dict' in checkpoint or 'discriminator' in checkpoint
            return has_generator or has_discriminator
        except:
            return False
    
    def initialize_gan(self):
        """Инициализация GAN модели"""
        try:
            from backend.gan.config import GANConfig  # ← ПРОВЕРЬ ПУТЬ!
            from backend.gan.models import GAN        # ← ПРОВЕРЬ ПУТЬ!
            
            config = GANConfig()
            self.gan_model = GAN(config)      # ← ТЕПЕРЬ ПРАВИЛЬНЫЙ КЛАСС
            self.current_status = "initialized"
            self._load_available_checkpoints()
            return True
        except Exception as e:
            print(f"Error initializing GAN: {e}")
            self.current_status = f"error: {str(e)}"
            return False
    
    def train_gan(self, real_data: pd.DataFrame, epochs: int = 50) -> Dict[str, Any]:
        """Обучение GAN модели с обновлением прогресса"""
        try:
            if self.gan_model is None:
                if not self.initialize_gan():
                    return {"success": False, "error": "Failed to initialize GAN"}
            
            self.current_status = "training_0%"
            self.training_progress = 0
            self.current_epoch = 0
            self.total_epochs = epochs
            
            # Сохраняем оригинальные методы
            original_train = self.gan_model.train
            original_validate = getattr(self.gan_model, '_validate_training', None)
            
            # Создаем обертку для отслеживания прогресса
            def training_wrapper(real_data, epochs=None):
                actual_epochs = epochs or self.gan_model.config.EPOCHS
                self.total_epochs = actual_epochs
                
                # Переопределяем валидацию для отслеживания прогресса
                def validate_with_progress(real_data, epoch):
                    self.current_epoch = epoch
                    progress = min(100, int((epoch / actual_epochs) * 100))
                    self.training_progress = progress
                    self.current_status = f"training_{progress}% (эпоха {epoch}/{actual_epochs})"
                    
                    print(f"Прогресс обучения: {progress}% (эпоха {epoch}/{actual_epochs})")
                    
                    if original_validate:
                        return original_validate(real_data, epoch)
                    return float('inf')
                
                self.gan_model._validate_training = validate_with_progress
                
                # Запускаем обучение
                result = original_train(real_data, epochs)
                
                # Завершение обучения
                self.training_progress = 100
                self.current_status = "trained"
                self.is_trained = True
                self._load_available_checkpoints()
                
                return result
            
            # Запускаем в отдельном потоке
            import threading
            thread = threading.Thread(target=training_wrapper, args=(real_data, epochs))
            thread.daemon = True
            thread.start()
            
            return {
                "success": True,
                "status": "training_started",
                "message": f"Обучение запущено на {epochs} эпох"
            }
            
        except Exception as e:
            self.current_status = f"error: {str(e)}"
            return {"success": False, "error": str(e)}
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> Optional[pd.DataFrame]:
        """Генерация синтетических данных"""
        try:
            if self.gan_model is None or not self.is_trained:
                return None
            
            self.current_status = "generating"
            synthetic_data = self.gan_model.generate(num_samples)
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
            "checkpoints": self.available_checkpoints[:10]  # Только последние 10
        }
        
        if self.gan_model:
            g_losses = self.gan_model.g_losses or []
            d_losses = self.gan_model.d_losses or []
            
            status_info["loss_history"] = {
                "g_losses": g_losses[-10:],
                "d_losses": d_losses[-10:],
                "total_epochs": len(g_losses),
                "latest_g_loss": g_losses[-1] if g_losses else None,
                "latest_d_loss": d_losses[-1] if d_losses else None
            }
            
            if hasattr(self.gan_model, 'wasserstein_distances') and self.gan_model.wasserstein_distances:
                wasserstein = self.gan_model.wasserstein_distances
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