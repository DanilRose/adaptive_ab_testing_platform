from __future__ import annotations

from datetime import datetime
import os
import tempfile
from typing import Any, Optional

import torch
from fastapi import BackgroundTasks

from backend.microservices.gan.config import GANConfig
from backend.microservices.data_gan.service import CheckpointStorageService


class DataGANLifecycleService:
    """Оркестрация lifecycle операций GAN (обучение/чекпоинты) для API-слоя."""

    @staticmethod
    def build_effective_config(*, epochs: int, gan_config: Optional[dict[str, Any]]) -> dict[str, Any]:
        base_config = GANConfig()
        effective_config: dict[str, Any] = {}
        for key, value in base_config.__dict__.items():
            if isinstance(value, torch.device):
                effective_config[key] = str(value)
            else:
                effective_config[key] = value

        if gan_config:
            for key, value in gan_config.items():
                if key in effective_config:
                    effective_config[key] = value

        effective_config["EPOCHS"] = epochs
        return effective_config

    @staticmethod
    def enqueue_training(
        *,
        background_tasks: BackgroundTasks,
        epochs: int,
        real_data_samples: int,
        save_checkpoint: bool,
        checkpoint_name: Optional[str],
        gan_config: Optional[dict[str, Any]],
        trained_by: str,
        data_generator: Any,
        gan_service: Any,
        status_cache: Any,
    ) -> None:
        effective_config = DataGANLifecycleService.build_effective_config(
            epochs=epochs,
            gan_config=gan_config,
        )

        def train_in_background(username: str, checkpoint_name_override: Optional[str]):
            try:
                real_data = data_generator.generate_dataset(real_data_samples)
                gan_service.train_gan(
                    real_data,
                    epochs,
                    config_overrides=effective_config,
                )

                if save_checkpoint and gan_service.gan_model and gan_service.is_trained:
                    checkpoint_payload = {
                        "epoch": gan_service.current_epoch,
                        "generator_state_dict": gan_service.gan_model.generator.state_dict(),
                        "discriminator_state_dict": gan_service.gan_model.discriminator.state_dict(),
                        "optimizer_G_state_dict": gan_service.gan_model.optimizer_G.state_dict(),
                        "optimizer_D_state_dict": gan_service.gan_model.optimizer_D.state_dict(),
                        "g_losses": gan_service.gan_model.g_losses,
                        "d_losses": gan_service.gan_model.d_losses,
                        "feature_info": gan_service.gan_model.feature_info,
                        "processed_columns": gan_service.gan_model.processed_columns,
                        "scalers": gan_service.gan_model.scalers,
                    }

                    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
                        torch.save(checkpoint_payload, tmp_file.name)
                        tmp_file_path = tmp_file.name

                    with open(tmp_file_path, "rb") as f:
                        checkpoint_bytes = f.read()
                    os.unlink(tmp_file_path)

                    generated_checkpoint_name = checkpoint_name_override or f"gan_trained_{datetime.now().strftime('%Y%m%d_%H%M')}.pth"
                    CheckpointStorageService.save_checkpoint_bytes_with_local_session(
                        checkpoint_name=generated_checkpoint_name,
                        checkpoint_bytes=checkpoint_bytes,
                        epoch=gan_service.current_epoch,
                        trained_by=username,
                        final_g_loss=gan_service.gan_model.g_losses[-1] if gan_service.gan_model.g_losses else None,
                        final_d_loss=gan_service.gan_model.d_losses[-1] if gan_service.gan_model.d_losses else None,
                    )

                    checkpoint_name_clean = generated_checkpoint_name[:-4] if generated_checkpoint_name.endswith('.pth') else generated_checkpoint_name
                    gan_service.loaded_checkpoint_name = checkpoint_name_clean
                    gan_service.current_status = "checkpoint_loaded"

                    status_cache.invalidate("gan_checkpoints")
                    status_cache.invalidate("gan_status")
            except Exception:
                import traceback
                traceback.print_exc()

        background_tasks.add_task(train_in_background, trained_by, checkpoint_name)
