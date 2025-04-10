import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from configs import CFG, InjectionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LayoutModelConfig:
    model_dir: Path
    max_height: int = field(default=640)
    max_weight: int = field(default=640)
    device: str = field(default="cpu")
    threshold: float = field(default=0.3)
    batch_size: int = field(default=1)

    @classmethod
    def init_from_cfg(cls, cfg: InjectionConfig = CFG) -> "LayoutModelConfig":
        return LayoutModelConfig(
            model_dir=cfg.LAYOUT_MODEL_PATH,
            max_weight=cfg.LAYOUTMAX_WEIDTH,
            max_height=cfg.LAYOUTM_HEIGHT,
            device=cfg.LAYOUTM_DEVICE,
            threshold=cfg.LAYOUTM_THRESHOLD,
            batch_size=cfg.LAYOUTM_BATCH_SIZE,
        )

    @property
    def preprocessor_config(self) -> str:
        preprocessor_cnf_path = self.model_dir.joinpath(*["model_artifacts", "layout", "preprocessor_config.json"])
        return str(preprocessor_cnf_path)

    @property
    def model_config(self) -> str:
        model_cnf_path = self.model_dir.joinpath(*["model_artifacts", "layout", "config.json"])
        return str(model_cnf_path)

    @property
    def artifact_path(self) -> str:
        artifact_path = self.model_dir.joinpath(*["model_artifacts", "layout"])
        return str(artifact_path)


@dataclass(frozen=True, slots=True, unsafe_hash=True)
class BoundingBox:
    width: int
    height: int
    left: float
    top: float
    right: float
    bottom: float
    score: float
    label: int
    image_name: str


class LayoutModel:
    """
    This class is used to create a layout for Hugging Face models.
    """

    cfg: LayoutModelConfig

    def __init__(self, cfg: LayoutModelConfig):
        """
        Initialize the HuggingFaceLayout class.

        Args:
            model_name (str): The name of the Hugging Face model.
        """
        self.cfg = cfg

    @classmethod
    def init_from_app_cfg(cls, cfg: InjectionConfig) -> "LayoutModel":
        config_ = LayoutModelConfig.init_from_cfg(cfg)
        return cls(config_)

    @classmethod
    def init_from_cfg(cls, config: LayoutModelConfig) -> "LayoutModel":
        return cls(config)

    def _process_batch(self, images_path: list[Path]) -> list[BoundingBox]:
        preprocessor = RTDetrImageProcessor.from_json_file(self.cfg.preprocessor_config)
        model = RTDetrForObjectDetection.from_pretrained(self.cfg.artifact_path, config=self.cfg.model_config).to_empty(
            device=self.cfg.device
        )
        model = model.to(self.cfg.device)
        resize_cng = {"height": self.cfg.max_weight, "width": self.cfg.max_height}
        images = [Image.open(image_path) for image_path in images_path]
        logger.info(f"Processing {len(images)} images ...")
        inputs = preprocessor(images, return_tensors="pt", size=resize_cng).to(self.cfg.device)
        outputs = model(**inputs)
        results = preprocessor.post_process_object_detection(
            outputs, target_sizes=[img.size[::-1] for img in images], threshold=self.cfg.threshold
        )

        bounding_boxes = []
        for idx, result in enumerate(results):
            score_ = result["scores"]
            label_ = result["labels"]
            box_ = result["boxes"]
            width, height = images[idx].size
            for score_t, label_t, box_t in zip(score_, label_, box_):
                bbox = [float(b) for b in box_t]
                bounding_boxes.append(
                    BoundingBox(
                        width=width,
                        height=height,
                        left=min(width, max(0, bbox[0])),
                        top=min(height, max(0, bbox[1])),
                        right=min(width, max(0, bbox[2])),
                        bottom=min(height, max(0, bbox[3])),
                        score=round(score_t.item(), 4),
                        label=int(label_t.item()) + 1,
                        image_name=images_path[idx].name,
                    )
                )
        return bounding_boxes

    def extract(self, image_path: str | Path) -> list[BoundingBox]:
        """
        Create the layout for the Hugging Face model.
        """
        # Implementation for creating the layout goes here

        pdf_files = list(Path(image_path).glob("*.png"))
        page_count = len(pdf_files)
        batch_size = self.cfg.batch_size
        # Split the PDF files into batches
        batches = [pdf_files[i : i + batch_size] for i in range(0, page_count, batch_size)]
        all_bounding_boxes = []
        with ThreadPoolExecutor(max_workers=min(len(batches), batch_size)) as executor:
            futures = [executor.submit(self._process_batch, batch) for batch in batches]
            for future in futures:
                try:
                    # Accumulate results from each future
                    all_bounding_boxes.extend(future.result())
                except Exception as e:  # noqa: PERF203
                    logger.info(f"Error processing batch: {e}")
        return all_bounding_boxes
