"""Detection comparison utilities."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..logger import get_logger
from ..models.inference import Detection


@dataclass
class ComparisonResult:
    """Result of comparing two sets of detections."""
    matched_pairs: List[Tuple[Detection, Detection, float]]  # (det1, det2, iou)
    unmatched_first: List[Detection]
    unmatched_second: List[Detection]
    mean_iou: Optional[float]
    min_iou: Optional[float]
    max_iou: Optional[float]


class DetectionComparator:
    """Compare detections from different models."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize comparator.
        
        Args:
            iou_threshold: Minimum IoU to consider a match
        """
        self.logger = get_logger()
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return float(intersection / union) if union > 0 else 0.0
    
    def compare(
        self,
        detections_first: List[Detection],
        detections_second: List[Detection],
    ) -> ComparisonResult:
        """
        Compare two sets of detections using IoU matching.
        
        Args:
            detections_first: First set of detections (e.g., PyTorch)
            detections_second: Second set of detections (e.g., ONNX)
            
        Returns:
            ComparisonResult with matched pairs and statistics
        """
        self.logger.info(
            f"Comparing {len(detections_first)} vs {len(detections_second)} detections"
        )
        
        matched_pairs = []
        used_second = set()
        
        for det1 in detections_first:
            best_iou = 0.0
            best_match = None
            best_idx = -1
            
            for idx, det2 in enumerate(detections_second):
                if idx in used_second:
                    continue
                if det1.class_id != det2.class_id:
                    continue
                
                iou = self.calculate_iou(det1.box, det2.box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = det2
                    best_idx = idx
            
            if best_iou >= self.iou_threshold and best_match is not None:
                matched_pairs.append((det1, best_match, best_iou))
                used_second.add(best_idx)
        
        matched_first_indices = set(range(len(matched_pairs)))
        unmatched_first = [
            d for i, d in enumerate(detections_first)
            if i >= len(matched_pairs) or i not in matched_first_indices
        ]
        unmatched_first = [
            d for d in detections_first
            if not any(d is m[0] for m in matched_pairs)
        ]
        unmatched_second = [
            d for i, d in enumerate(detections_second)
            if i not in used_second
        ]
        
        ious = [m[2] for m in matched_pairs]
        
        result = ComparisonResult(
            matched_pairs=matched_pairs,
            unmatched_first=unmatched_first,
            unmatched_second=unmatched_second,
            mean_iou=float(np.mean(ious)) if ious else None,
            min_iou=float(np.min(ious)) if ious else None,
            max_iou=float(np.max(ious)) if ious else None,
        )
        
        self.logger.info(f"Matched: {len(matched_pairs)}, Unmatched first: {len(unmatched_first)}, Unmatched second: {len(unmatched_second)}")
        if result.mean_iou:
            self.logger.info(f"IoU stats - Mean: {result.mean_iou:.4f}, Min: {result.min_iou:.4f}, Max: {result.max_iou:.4f}")
        
        return result
    
    def print_report(self, result: ComparisonResult, first_name: str = "First", second_name: str = "Second") -> None:
        """Print a formatted comparison report."""
        print("\n" + "=" * 60)
        print(f"DETECTION COMPARISON: {first_name} vs {second_name}")
        print("=" * 60)
        
        print(f"\nMatched detections (IoU >= {self.iou_threshold}): {len(result.matched_pairs)}")
        print(f"{first_name} only: {len(result.unmatched_first)}")
        print(f"{second_name} only: {len(result.unmatched_second)}")
        
        if result.mean_iou is not None:
            print(f"\nIoU Statistics:")
            print(f"  Mean IoU: {result.mean_iou:.4f}")
            print(f"  Min IoU:  {result.min_iou:.4f}")
            print(f"  Max IoU:  {result.max_iou:.4f}")
            
            print(f"\nDetailed matches:")
            for i, (det1, det2, iou) in enumerate(result.matched_pairs):
                print(f"  {i+1}. [{det1.class_name}] IoU={iou:.4f}")
