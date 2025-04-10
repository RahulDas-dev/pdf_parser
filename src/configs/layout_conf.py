from pathlib import Path
from typing import Literal

from pydantic import DirectoryPath, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings

DEVICETYPE = Literal["cpu", "gpu", "auto"]


class LayoutAnalyzerConfig(BaseSettings):
    LAYOUT_MODEL_PATH: DirectoryPath = Field(description="Path to the layout model")
    LAYOUTM_HEIGHT: PositiveInt = Field(default=640)
    LAYOUTMAX_WEIDTH: PositiveInt = Field(default=640)
    LAYOUTM_DEVICE: DEVICETYPE = Field(default="cpu")
    LAYOUTM_THRESHOLD: float = Field(default=0.3)
    LAYOUTM_BATCH_SIZE: PositiveInt = Field(default=1)

    @field_validator("LAYOUT_MODEL_PATH", mode="before")
    @classmethod
    def validate_model_path(cls, v: DirectoryPath) -> DirectoryPath:
        if not Path(v).is_dir():
            raise ValueError(f"Layout model path {v} does not exist")
        atrifact_path = Path(v) / "model_artifacts"
        if not atrifact_path.is_dir():
            raise ValueError(f"Layout model artifacts path {atrifact_path} does not exist")
        layout_model_path = atrifact_path / "layout"
        if not layout_model_path.is_dir():
            raise ValueError(f"Layout model path {layout_model_path} does not exist")
        model_config_path = layout_model_path / "config.json"
        if not model_config_path.is_file():
            raise ValueError(f"Layout model config path {model_config_path} does not exist")
        preprocessor_config_path = layout_model_path / "preprocessor_config.json"
        if not preprocessor_config_path.is_file():
            raise ValueError(f"Layout model preprocessor config path {preprocessor_config_path} does not exist")
        model_safe_tensor = layout_model_path / "model.safetensors"
        if not model_safe_tensor.is_file():
            raise ValueError(f"Layout model safe tensor path {model_safe_tensor} does not exist")
        return v
