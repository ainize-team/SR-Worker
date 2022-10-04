from typing import Optional

from pydantic import BaseModel, HttpUrl

from enums import ModelEnum, ResponseStatusEnum


class Images(BaseModel):
    input: HttpUrl
    output: Optional[HttpUrl]


class Error(BaseModel):
    status_code: int
    error_message: str


class SuperResolutionResult(BaseModel):
    user_id: str
    model_name: ModelEnum
    images: Images
    status: ResponseStatusEnum = ResponseStatusEnum.PENDING
    error: Optional[Error] = None
    updated_at: int = 0
