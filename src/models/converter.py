"""ONNX model conversion module."""

import shutil
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from ..logger import get_logger


class ONNXConverter:
    """Handles conversion of YOLO models to ONNX format."""
    
    def __init__(self, pytorch_model_path: Path):
        """
        Initialize converter.
        
        Args:
            pytorch_model_path: Path to PyTorch YOLO model
        """
        self.logger = get_logger()
        self.pytorch_model_path = Path(pytorch_model_path)
        
        if not self.pytorch_model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.pytorch_model_path}")
        
        if self.pytorch_model_path.suffix != ".pt":
            raise ValueError(f"Expected .pt file, got: {self.pytorch_model_path.suffix}")
    
    def convert(
        self,
        output_path: Optional[Path] = None,
        input_size: int = 640,
        simplify: bool = True,
        dynamic: bool = False,
        half: bool = False,
    ) -> Path:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            output_path: Desired output path for ONNX model
            input_size: Input image size
            simplify: Whether to simplify the ONNX model
            dynamic: Whether to use dynamic input shapes
            half: Whether to use FP16 precision
            
        Returns:
            Path to the exported ONNX model
        """
        self.logger.info(f"Converting model to ONNX: {self.pytorch_model_path}")
        self.logger.info(f"Parameters: imgsz={input_size}, simplify={simplify}, dynamic={dynamic}, half={half}")
        
        model = YOLO(str(self.pytorch_model_path))
        
        export_path = model.export(
            format="onnx",
            imgsz=input_size,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
        )
        
        export_path = Path(export_path)
        self.logger.info(f"Model exported to: {export_path}")
        
        if output_path and export_path != Path(output_path):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(export_path), str(output_path))
            self.logger.info(f"Moved to: {output_path}")
            return output_path
        
        return export_path
    
    def validate_onnx(self, onnx_path: Path) -> bool:
        """
        Validate ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            True if model is valid
        """
        import onnx
        
        self.logger.info(f"Validating ONNX model: {onnx_path}")
        
        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            self.logger.info("ONNX model validation passed")
            return True
        except Exception as e:
            self.logger.error(f"ONNX validation failed: {e}")
            return False
