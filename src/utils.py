import gc
import os
from datetime import datetime

import cv2
import numpy as np
import requests
import torch
from firebase_admin import db, storage
from pydantic import HttpUrl

from configs.config import firebase_settings
from enums import ErrorStatusEnum
from schemas import Error

app_name = firebase_settings.firebase_app_name


def get_now_timestamp() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def download_image_from_storage(task_id: str, url: str) -> np.ndarray:
    res = requests.get(url)
    image_path = f"{task_id}/input.png"
    os.makedirs(task_id, exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(res.content)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    return image


def save_image_to_storage(task_id: str, image_path: str) -> HttpUrl:
    bucket = storage.bucket()
    base_name = os.path.basename(image_path)

    blob = bucket.blob(f"{app_name}/results/{task_id}/{base_name}")
    blob.upload_from_filename(image_path)
    blob.make_public()

    url = blob.public_url

    return url


def save_error(task_id: str, status_code: ErrorStatusEnum, error_message: str):
    error = Error(status_code=status_code, error_message=error_message)
    db.reference(f"{app_name}/{task_id}").update(
        {
            "error": error,
            "updated_at": get_now_timestamp(),
        }
    )


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
