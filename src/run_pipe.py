# ruff: noqa: LOG015
# import sys

# sys.path.append("src")
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from configs import CFG, InjectionConfig
from pdftoimg.pdfinum import PdfiumBackend

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def process_pdf(pdf_path: Path, config: dict[str, Any]) -> tuple[str, int]:
    """Process a single PDF file."""
    config_ = InjectionConfig(**config)
    backend = PdfiumBackend.init_from_app_cfg(config_)
    return backend.convert(pdf_path)


def main() -> None:
    input_directory = Path(CFG.INPUT_DIRECTORY)
    pdf_files = list(input_directory.glob("*.pdf"))  # Collect all PDF files in the input directory

    if not pdf_files:
        logging.info("No PDF files found in the input directory.")
        return

    logging.info(f"Found {len(pdf_files)} PDF files. Starting processing...")

    start_time = time.perf_counter()

    # Use ProcessPoolExecutor for multiprocessing
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_pdf, pdf, CFG.model_dump()): pdf for pdf in pdf_files}

        for future in as_completed(futures):
            try:
                output_folder, page_count = future.result()
                logging.info(f"Processed {page_count} pages. Output saved to {output_folder}")
            except Exception as e:
                logging.error(f"Error processing file {futures[future]}: {e}")

    logging.info(f"Total time taken: {time.perf_counter() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
