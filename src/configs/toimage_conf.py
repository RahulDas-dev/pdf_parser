import os

from pydantic import DirectoryPath, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings


class ToImageConfig(BaseSettings):
    POPPLER_PATH: DirectoryPath | None = Field(description="Path to the POPPLER binary", default=None)
    INPUT_DIRECTORY: DirectoryPath = Field(description="Path to the propeller model")
    ALLOWED_EXTENSIONS: list[str] = Field(
        description="Allowed extensions for the uploaded files",
        default_factory=lambda: ["pdf", "PDF", "png", "PNG"],
    )
    MAX_IMG_WIDTH: int = Field(description="Maximum image width", default=-1)
    MAX_IMG_HEIGHT: int = Field(description="Maximum image height", default=-1)
    SAVE_FORMAT: str = Field(default="PNG")
    PDF2IMG_THREAD_COUNT: PositiveInt = Field(default=1)

    @field_validator("PDF2IMG_THREAD_COUNT", mode="before")
    @classmethod
    def validate_thread_count(cls, v: int) -> int | None:
        if v in (-1, "-1"):
            cpu_count_ = os.cpu_count()
            v = cpu_count_ * 2 if cpu_count_ is not None else 1
        return v
