import hashlib
import os

import requests
import torch
from loguru import logger

from configs.config import model_settings
from constants import MODEL_INFO
from enums import DeviceEnum, ExitCodeEnum, ModelEnum
from swinir import SwinIR


def get_hash(model_path: str) -> str:
    with open(model_path, "rb") as f:
        data = f.read()
        model_hash = hashlib.sha256(data).hexdigest()
    return model_hash


def download_model(model_url: str, model_path: str) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    res = requests.get(model_url, allow_redirects=True)
    if res.status_code == 200:
        model_hash = hashlib.sha256(res.content).hexdigest()
        with open(model_path, "wb") as f:
            f.write(res.content)
        return model_hash
    else:
        raise requests.exceptions.RequestException(
            f"Model download error: {res.status_code}"
        )


def define_model(model_name: ModelEnum, model_path: str) -> SwinIR:
    # 003 real-world image sr
    if model_name == ModelEnum.SWIN_LR_X4:
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="1conv",
        )
        param_key_g = "params_ema"
    elif model_name == ModelEnum.SWIN_LR_LARGE_X4:
        model = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=240,
            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
            mlp_ratio=2,
            upsampler="nearest+conv",
            resi_connection="3conv",
        )
        param_key_g = "params_ema"

    pretrained_model = torch.load(model_path)
    model.load_state_dict(
        pretrained_model[param_key_g]
        if param_key_g in pretrained_model.keys()
        else pretrained_model,
        strict=True,
    )
    model.eval()

    return model


class SuperResolutionModel:
    def __init__(self):
        self._load_model()

    def _load_model(self):
        model_name = model_settings.model_name
        model_path = model_settings.model_path

        # if model path is not valid, try to download model from url
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            logger.warning(f"{model_path} is not valid path, try to download model")
            try:
                model_hash = download_model(
                    MODEL_INFO[model_name].model_url, model_path
                )
            except Exception as e:
                logger.error(f"Error : {e}")
                exit(ExitCodeEnum.MODEL_DOWNLOAD_ERROR)
        else:
            model_hash = get_hash(model_path)
        logger.info("Check sha256 value")
        if model_hash != MODEL_INFO[model_name].sha_256:
            logger.error(
                f"Sha256 value({model_hash}) is not valid, try to download model"
            )
            try:
                model_hash = download_model(
                    MODEL_INFO[model_name].model_url, model_path
                )
            except Exception as e:
                logger.error(f"Error : {e}")
                exit(ExitCodeEnum.MODEL_DOWNLOAD_ERROR)
            if model_hash != MODEL_INFO[model_name].sha_256:
                logger.error(f"Sha256 value({model_hash}) is not valid.")
                exit(ExitCodeEnum.MODEL_CHECKSUM_ERROR)
        self.model = define_model(model_name, model_path)
        self.model.to(model_settings.device)
        if model_settings.device == DeviceEnum.CUDA:
            self.model.half()
        logger.info("The model was successfully loaded.")

    def upscale(self, x):
        output = self.model(x)

        return output
