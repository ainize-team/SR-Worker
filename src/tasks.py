import shutil
from functools import partialmethod

import cv2
import numpy as np
import torch
from celery.signals import celeryd_init
from firebase_admin import db
from loguru import logger
from torch import autocast
from tqdm import tqdm

from configs.config import firebase_settings, model_settings
from constants import MODEL_INFO
from enums import DeviceEnum, ErrorStatusEnum, ResponseStatusEnum
from models import SuperResolutionModel
from utils import (
    clear_memory,
    delete_image_from_storage,
    download_image_from_storage,
    get_now_timestamp,
    save_error,
    save_image_to_storage,
)
from worker import app

sr = SuperResolutionModel()


@celeryd_init.connect
def load_model(**kwargs):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    logger.info("Start loading model...")
    sr.load_model()
    logger.info("Loading model is done!")


@app.task(name="upscale")
def upscale(task_id: str):
    def after_inference(task_id: str):
        clear_memory()
        shutil.rmtree(task_id, ignore_errors=True)

    app_name = firebase_settings.firebase_app_name
    db.reference(f"{app_name}/{task_id}").update(
        {
            "status": ResponseStatusEnum.ASSIGNED,
            "model_name": model_settings.model_name,
            "updated_at": get_now_timestamp(),
        }
    )

    try:
        scale = MODEL_INFO[model_settings.model_name].scale
        img_lq = download_image_from_storage(task_id, "input.png")
        img_lq = img_lq.astype(np.float32) / 255.0
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0)  # CHW-RGB to NCHW-RGB
        img_lq = img_lq.to(model_settings.device)
        if model_settings.device == DeviceEnum.CUDA:
            img_lq = img_lq.half()
            # inference
            with autocast("cuda"):
                with torch.no_grad():
                    # pad input image to be a multiple of window_size
                    _, _, h_old, w_old = img_lq.size()
                    output = sr.upscale(img_lq)
                    output = output[..., : h_old * scale, : w_old * scale]
        else:
            with torch.no_grad():
                # pad input image to be a multiple of window_size
                _, _, h_old, w_old = img_lq.size()
                output = sr.upscale(img_lq)
                output = output[..., : h_old * scale, : w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output_path = f"{task_id}/output.png"
        cv2.imwrite(output_path, output)
        output_url = save_image_to_storage(task_id, output_path)
        delete_image_from_storage(task_id, "input.png")

        db.reference(f"{app_name}/{task_id}").update(
            {
                "status": ResponseStatusEnum.COMPLETED,
                "updated_at": get_now_timestamp(),
                "images": {"output": output_url},
            }
        )
    except ValueError as e:
        save_error(task_id, ErrorStatusEnum.UNPROCESSABLE_ENTITY, str(e))
    except Exception as e:
        save_error(task_id, ErrorStatusEnum.INTERNAL_SERVER_ERROR, str(e))
    finally:
        after_inference(task_id)
