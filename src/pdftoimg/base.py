from pathlib import Path
from typing import Protocol, runtime_checkable

from configs.main import InjectionConfig


@runtime_checkable
class Pdf2ImageBackend(Protocol):
    def __init__(self, config: InjectionConfig):
        """Initialize with the path to the PDF."""
        ...

    def convert(self, pdf_path: str | Path) -> str:
        """Process a specific page number."""
        ...
