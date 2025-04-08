import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pdf2image import convert_from_bytes
from PIL.Image import Image

from configs import InjectionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Pdf2ImgConfig:
    poppler_path: str
    output_path: Path = field(default_factory=Path)
    max_width: int = field(default=-1)
    max_height: int = field(default=-1)
    thread_count: int = field(default=1)
    save_format: str = field(default="png")

    @property
    def resize_ops_enabled(self) -> bool:
        return self.max_width > 0 and self.max_height > 0

    @classmethod
    def init_from_cfg(cls, cfg: InjectionConfig) -> "Pdf2ImgConfig":
        poppler_path_ = cfg.POPPLER_PATH
        if poppler_path_ is None or poppler_path_ == "":
            raise ValueError("Poppler path is not set")
        output_path_ = Path(cfg.OUTPUT_DIRECTORY) / Path("pdf2img")
        if not output_path_.exists():
            output_path_.mkdir(parents=True)
        thread_count_ = 1 if cfg.PDF2IMG_THREAD_COUNT is None else cfg.PDF2IMG_THREAD_COUNT
        return Pdf2ImgConfig(
            poppler_path=str(poppler_path_),
            output_path=output_path_,
            max_width=cfg.MAX_IMG_WIDTH,
            max_height=cfg.MAX_IMG_HEIGHT,
            thread_count=thread_count_,
            save_format=cfg.SAVE_FORMAT,
        )


class Pdf2ImgBackend:
    cfg: Pdf2ImgConfig

    def __init__(self, config: Pdf2ImgConfig):
        self.cfg = config

    @classmethod
    def init_from_app_cfg(cls, cfg: InjectionConfig) -> "Pdf2ImgBackend":
        config_ = Pdf2ImgConfig.init_from_cfg(cfg)
        return cls(config_)

    @classmethod
    def init_from_cfg(cls, config: Pdf2ImgConfig) -> "Pdf2ImgBackend":
        return cls(config)

    def _convert_to_image_and_save(self, page_img: Image, page_index: int, output_folder: Path) -> None:
        save_format_ = "PNG" if self.cfg.save_format == "png" else self.cfg.save_format
        save_path = output_folder / f"Page_{page_index:02}.png"
        if not self.cfg.resize_ops_enabled:
            logger.info(f"processing Page No {page_index} Images shape {page_img.size} ...")
            page_img.save(save_path, save_format_)
            return
        width, height = page_img.size
        if width < height and height > self.cfg.max_height:
            new_height = self.cfg.max_height
            new_width = int(new_height * (width / height))
            new_image = page_img.resize((new_width, new_height))
            logger.info(f"processing Page No {page_index} Images shape {new_image.size} ...")
            new_image.save(save_path, save_format_)
        elif width > height and width > self.cfg.max_width:
            new_width = self.cfg.max_width
            new_height = int(new_width * (height / width))
            new_image = page_img.resize((new_width, new_height))
            logger.info(f"processing Page No {page_index} Images shape {new_image.size} ...")
            new_image.save(save_path, save_format_)

    def _resolve_conflict(self, subfolder: str) -> str:
        if not (self.cfg.output_path / Path(subfolder)).exists():
            return subfolder
        count = 0
        while True:
            count += 1
            if not (self.cfg.output_path / Path(f"{subfolder}_{count}")).exists():
                return f"{subfolder}_{count}"

    def convert(self, pdf_path: str | Path) -> tuple[str, int]:
        filename = Path(pdf_path).stem
        filename = self._resolve_conflict(filename)
        output_folder = self.cfg.output_path / Path(filename)
        output_folder.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "rb") as pdf_file:
            images = convert_from_bytes(
                pdf_file.read(),
                poppler_path=self.cfg.poppler_path,
                fmt=self.cfg.save_format,
                thread_count=self.cfg.thread_count,
            )
        page_count = len(images)
        logger.info(f"Pdf Document Page count {page_count} ")
        start_time = time.perf_counter()
        if self.cfg.thread_count in [0, 1, None]:
            for page_index in range(page_count):
                self._convert_to_image_and_save(images[page_index], page_index + 1, output_folder)
            logger.info(f"Time taken to convert {page_count} pages: {time.perf_counter() - start_time:.2f} seconds")
            return str(output_folder), page_count
        with ThreadPoolExecutor(max_workers=self.cfg.thread_count) as executor:
            futures = [
                executor.submit(
                    self._convert_to_image_and_save,
                    images[page_index],
                    page_index + 1,
                    output_folder,
                )
                for page_index in range(page_count)
            ]
            for future in futures:
                future.result()
        logger.info(f"Time taken to convert {page_count} pages: {time.perf_counter() - start_time:.2f} seconds")
        return str(output_folder), page_count
