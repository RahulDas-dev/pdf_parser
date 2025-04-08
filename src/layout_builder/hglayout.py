from dataclasses import dataclass, field
from pathlib import Path

from configs import CFG, InjectionConfig


@dataclass(frozen=True, slots=True)
class PdfiumConfig:
    max_width: int = field(default=-1)
    max_height: int = field(default=-1)
    thread_count: int = field(default=1)
    save_format: str = field(default="png")

    @property
    def resize_ops_enabled(self) -> bool:
        return self.max_width > 0 and self.max_height > 0

    @classmethod
    def init_from_cfg(cls, cfg: InjectionConfig = CFG) -> "PdfiumConfig":
        output_path_ = Path(cfg.OUTPUT_DIRECTORY) / Path("pdf2img")
        if not output_path_.exists():
            output_path_.mkdir(parents=True)
        return PdfiumConfig(
            max_width=cfg.MAX_IMG_WIDTH,
            max_height=cfg.MAX_IMG_HEIGHT,
            thread_count=cfg.PDF2IMG_THREAD_COUNT,
            save_format=cfg.SAVE_FORMAT,
        )


class HuggingFaceLayout:
    """
    This class is used to create a layout for Hugging Face models.
    """

    def __init__(self, model_name: str):
        """
        Initialize the HuggingFaceLayout class.

        Args:
            model_name (str): The name of the Hugging Face model.
        """
        self.model_name = model_name
        self.layout = None

    def create_layout(self):
        """
        Create the layout for the Hugging Face model.
        """
        # Implementation for creating the layout goes here
