import torch
from constants import MODEL_INFO
from enums import DeviceEnum, ModelEnum
from pydantic import BaseSettings


class CeleryWorkerSettings(BaseSettings):
    worker_name: str = "Celery Worker"
    broker_uri: str = "amqp://guest:guest@localhost:5672//"


class ModelSettings(BaseSettings):
    device: DeviceEnum = (
        DeviceEnum.CUDA if torch.cuda.is_available() else DeviceEnum.CPU
    )
    model_name: ModelEnum = ModelEnum.SWIN_LR_LARGE_X4
    model_path: str = f"./model/{MODEL_INFO[ModelEnum.SWIN_LR_LARGE_X4].file_name}"


class FirebaseSettings(BaseSettings):
    firebase_app_name: str = "super-resolution"
    cred_path: str = "/app/key/serviceAccountKey.json"
    database_url: str
    storage_bucket: str


celery_worker_settings = CeleryWorkerSettings()
model_settings = ModelSettings()
firebase_settings = FirebaseSettings()
