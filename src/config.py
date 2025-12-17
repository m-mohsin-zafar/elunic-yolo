"""Configuration management for the YOLO inference pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Model-related configuration."""
    pytorch_model: Path
    onnx_model: Path
    input_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45


@dataclass
class PathConfig:
    """Path-related configuration."""
    base_dir: Path
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    images_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.base_dir / "data"
        self.output_dir = self.base_dir / "output"
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.images_dir = self.data_dir / "images"


@dataclass
class Config:
    """Main configuration container."""
    paths: PathConfig
    model: ModelConfig
    
    @classmethod
    def from_defaults(cls, base_dir: Optional[Path] = None) -> "Config":
        """Create configuration with default values."""
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
        
        paths = PathConfig(base_dir=base_dir)
        
        model = ModelConfig(
            pytorch_model=paths.checkpoints_dir / "yolo11n.pt",
            onnx_model=paths.checkpoints_dir / "yolo11n.onnx",
        )
        
        return cls(paths=paths, model=model)
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.images_dir.mkdir(parents=True, exist_ok=True)
