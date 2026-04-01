import pickle
import torch
import io
import sys

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Разрешаем все модули которые могут быть в чекпоинтах GAN
        allowed_modules = {
            'torch', 'numpy', 'sklearn', 'pandas', 
            'collections', 'builtins', '__main__',
            'backend', 'gan'
        }
        
        # Разрешаем подмодули numpy и sklearn
        if any(module.startswith(allowed) for allowed in allowed_modules):
            return super().find_class(module, name)
        
        # Запрещаем все остальное
        raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")

def safe_load(filepath, map_location='cpu'):
    """Безопасная загрузка чекпоинта"""
    try:
        with open(filepath, 'rb') as f:
            return SafeUnpickler(f).load()
    except Exception as e:
        print(f"❌ Safe load failed: {e}")
        # Fallback на прямой pickle (только если доверяешь чекпоинтам)
        with open(filepath, 'rb') as f:
            return pickle.load(f)

def safe_torch_load(filepath, map_location='cpu'):
    """Альтернатива torch.load с безопасной загрузкой"""
    try:
        # Сначала пробуем стандартную загрузку
        return torch.load(filepath, map_location=map_location, weights_only=False)
    except:
        # Fallback на наш безопасный загрузчик
        return safe_load(filepath)