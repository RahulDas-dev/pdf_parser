from pathlib import Path
from typing import Protocol, runtime_checkable

from configs.main import InjectionConfig


@runtime_checkable
class BaseLayoutBuilder(Protocol):
    def __init__(self, config: InjectionConfig):
        """Initialize with the path to the PDF."""
        ...

    def extract(self, pdf_path: str | Path) -> str:
        """Process a specific page number."""
        ...
