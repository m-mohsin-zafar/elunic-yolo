"""YOLO model inference module."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from ..logger import get_logger


@dataclass
class Detection:
    """Single detection result."""
    box: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            "box": self.box.tolist(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


class YOLOInference:
    """YOLO model inference handler."""
    
    def __init__(self, model_path: Path):
        """
        Initialize YOLO inference.
        
        Args:
            model_path: Path to YOLO model (.pt or .onnx)
        """
        self.logger = get_logger()
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.logger.info(f"Loading model: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.model_type = "onnx" if self.model_path.suffix == ".onnx" else "pytorch"
        self.logger.info(f"Model loaded successfully (type: {self.model_type})")
    
    def predict(
        self,
        image_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """
        Run inference on an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of Detection objects
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.info(f"Running inference on: {image_path}")
        
        results = self.model(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
        )
        
        detections = self._parse_results(results)
        self.logger.info(f"Found {len(detections)} detections")
        
        return detections
    
    def _parse_results(self, results) -> List[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = Detection(
                    box=boxes.xyxy[i].cpu().numpy(),
                    confidence=float(boxes.conf[i].cpu().numpy()),
                    class_id=int(boxes.cls[i].cpu().numpy()),
                    class_name=result.names[int(boxes.cls[i].cpu().numpy())],
                )
                detections.append(detection)
        
        return detections
    
    def predict_and_visualize(
        self,
        image_path: Path,
        output_path: Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> List[Detection]:
        """
        Run inference and save annotated image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of Detection objects
        """
        results = self.model(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
        )
        
        detections = self._parse_results(results)
        
        annotated = results[0].plot()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        self.logger.info(f"Saved annotated image: {output_path}")
        
        return detections
