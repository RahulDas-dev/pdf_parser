import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium
from pypdfium2._helpers import PdfBitmap

from configs import InjectionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PdfiumConfig:
    output_path: Path = field(default_factory=Path)
    max_width: int = field(default=-1)
    max_height: int = field(default=-1)
    thread_count: int = field(default=1)
    save_format: str = field(default="png")

    @property
    def resize_ops_enabled(self) -> bool:
        return self.max_width > 0 and self.max_height > 0

    @classmethod
    def init_from_cfg(cls, cfg: InjectionConfig) -> "PdfiumConfig":
        output_path_ = Path(cfg.OUTPUT_DIRECTORY) / Path("pdf2img")
        if not output_path_.exists():
            output_path_.mkdir(parents=True)
        return PdfiumConfig(
            output_path=output_path_,
            max_width=cfg.MAX_IMG_WIDTH,
            max_height=cfg.MAX_IMG_HEIGHT,
            thread_count=cfg.PDF2IMG_THREAD_COUNT,
            save_format=cfg.SAVE_FORMAT,
        )


class PdfiumBackend:
    cfg: PdfiumConfig

    def __init__(self, config: PdfiumConfig):
        self.cfg = config

    @classmethod
    def init_from_app_cfg(cls, cfg: InjectionConfig) -> "PdfiumBackend":
        config_ = PdfiumConfig.init_from_cfg(cfg)
        return cls(config_)

    @classmethod
    def init_from_cfg(cls, config: PdfiumConfig) -> "PdfiumBackend":
        return cls(config)

    def _convert_to_image_and_save(self, page_bitmap: PdfBitmap, page_index: int, output_folder: Path) -> None:
        pil_image = page_bitmap.to_pil()

        save_format_ = "PNG" if self.cfg.save_format == "png" else self.cfg.save_format
        save_path = output_folder / f"Page_{page_index:04}.png"
        if not self.cfg.resize_ops_enabled:
            logger.info(f"processing Page No {page_index} Images shape {pil_image.size} ...")
            pil_image.save(save_path, save_format_)
            return
        width, height = pil_image.size
        if width < height and height > self.cfg.max_height:
            new_height = self.cfg.max_height
            new_width = int(new_height * (width / height))
            new_image = pil_image.resize((new_width, new_height))
            logger.info(f"processing Page No {page_index} Images shape {new_image.size} ...")
            new_image.save(save_path, save_format_)
        elif width > height and width > self.cfg.max_width:
            new_width = self.cfg.max_width
            new_height = int(new_width * (height / width))
            new_image = pil_image.resize((new_width, new_height))
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
        logger.info(f"Output Folder {output_folder} ")
        pdf_doc = pdfium.PdfDocument(pdf_path)
        page_count = len(pdf_doc)
        logger.info(f"Pdf Document Page count {page_count} ")
        start_time = time.perf_counter()
        if self.cfg.thread_count in [0, 1, None]:
            for page_index in range(page_count):
                self._convert_to_image_and_save(
                    pdf_doc[page_index].render(scale=2, rotation=0), page_index + 1, output_folder
                )
            logger.info(f"Time taken to convert {page_count} pages: {time.perf_counter() - start_time:.2f} seconds")
            return str(output_folder), page_count
        with ThreadPoolExecutor(max_workers=self.cfg.thread_count) as executor:
            futures = [
                executor.submit(
                    self._convert_to_image_and_save,
                    pdf_doc[page_index].render(scale=2, rotation=0),
                    page_index + 1,
                    output_folder,
                )
                for page_index in range(page_count)
            ]
            for future in futures:
                future.result()
        logger.info(f"Time taken to convert {page_count} pages: {time.perf_counter() - start_time:.2f} seconds")
        return str(output_folder), page_count
