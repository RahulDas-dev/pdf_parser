# ruff: noqa: LOG015
# import sys

# sys.path.append("src")
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from configs import CFG, InjectionConfig
from layout_builder.hglayout import LayoutModel
from pdftoimg.pdfinum import PdfiumBackend

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def process_pdf(pdf_path: Path, config: dict[str, Any]) -> tuple[str, int]:
    """Process a single PDF file."""
    config_ = InjectionConfig(**config)
    backend = PdfiumBackend.init_from_app_cfg(config_)
    return backend.convert(pdf_path)


def process_png(pdf_path: Path, config: dict[str, Any]) -> Any:
    """Process a single PDF file."""
    config_ = InjectionConfig(**config)
    backend = LayoutModel.init_from_app_cfg(config_)
    return backend.extract(pdf_path)


def convert_pdf2img() -> None:
    input_directory = Path(CFG.INPUT_DIRECTORY)
    pdf_files = list(input_directory.glob("*.pdf"))  # Collect all PDF files in the input directory

    if not pdf_files:
        logging.info("No PDF files found in the input directory.")
        return

    logging.info(f"Found {len(pdf_files)} PDF files. Starting processing...")

    start_time = time.perf_counter()
    if len(pdf_files) == 1:
        # If there's only one PDF file, process it directly
        output_folder, page_count = process_pdf(pdf_files[0], CFG.model_dump())
        logging.info(f"Processed {page_count} pages. Output saved to {output_folder}")
        return
    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, pdf, CFG.model_dump()): pdf for pdf in pdf_files}

        for future in as_completed(futures):
            try:
                output_folder, page_count = future.result()
                logging.info(f"Processed {page_count} pages. Output saved to {output_folder}")
            except Exception as e:  # noqa: PERF203
                logging.error(f"Error processing file {futures[future]}: {e}")

    logging.info(f"Total time taken: {time.perf_counter() - start_time:.2f} seconds")


def extarct_layout() -> None:
    input_directory = Path(CFG.OUTPUT_DIRECTORY) / "pdf2img"
    input_directorys = [p for p in input_directory.iterdir() if p.is_dir()]

    if not input_directorys:
        logging.info("No PDF files found in the input directory.")
        return
    page_imges_dir = input_directorys[0:1]
    logging.info(f"Found {len(page_imges_dir)} PDF files. Starting processing...")

    start_time = time.perf_counter()
    if len(page_imges_dir) == 1:
        # If there's only one PDF file, process it directly
        bboxes = process_png(page_imges_dir[0], CFG.model_dump())
        logging.info(f"Processed {len(bboxes)} pages. Output saved to {page_imges_dir[0]}")
        return
    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_png, pdf, CFG.model_dump()): pdf for pdf in page_imges_dir}

        for future in as_completed(futures):
            try:
                output_folder, page_count = future.result()
                logging.info(f"Processed {page_count} pages. Output saved to {output_folder}")
            except Exception as e:  # noqa: PERF203
                logging.error(f"Error processing file {futures[future]}: {e}")

    logging.info(f"Total time taken: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    extarct_layout()
