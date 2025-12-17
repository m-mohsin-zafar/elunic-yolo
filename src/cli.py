"""Command-line interface for the YOLO inference pipeline."""

import argparse
import sys
from pathlib import Path

from .config import Config
from .logger import setup_logger, get_logger
from .models import YOLOInference, ONNXConverter
from .utils import DetectionComparator


def print_detections(detections, title: str) -> None:
    """Print detections in a formatted way."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    
    for det in detections:
        box = det.box
        print(f"  [{det.class_name}] conf={det.confidence:.3f} box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
    
    print(f"\nTotal detections: {len(detections)}")


def run_pytorch_inference(args, config: Config) -> list:
    """Run PyTorch model inference."""
    logger = get_logger()
    
    model = YOLOInference(config.model.pytorch_model)
    output_path = config.paths.output_dir / "pytorch_result.png"
    
    detections = model.predict_and_visualize(
        image_path=args.image,
        output_path=output_path,
        conf_threshold=config.model.confidence_threshold,
        iou_threshold=config.model.iou_threshold,
    )
    
    print_detections(detections, "PYTORCH INFERENCE RESULTS")
    logger.info(f"Saved annotated image: {output_path}")
    
    return detections


def run_onnx_conversion(args, config: Config) -> Path:
    """Convert PyTorch model to ONNX."""
    converter = ONNXConverter(config.model.pytorch_model)
    
    onnx_path = converter.convert(
        output_path=config.model.onnx_model,
        input_size=config.model.input_size,
        simplify=True,
    )
    
    converter.validate_onnx(onnx_path)
    
    return onnx_path


def run_onnx_inference(args, config: Config, onnx_path: Path) -> list:
    """Run ONNX model inference."""
    logger = get_logger()
    
    model = YOLOInference(onnx_path)
    output_path = config.paths.output_dir / "onnx_result.png"
    
    detections = model.predict_and_visualize(
        image_path=args.image,
        output_path=output_path,
        conf_threshold=config.model.confidence_threshold,
        iou_threshold=config.model.iou_threshold,
    )
    
    print_detections(detections, "ONNX INFERENCE RESULTS")
    logger.info(f"Saved annotated image: {output_path}")
    
    return detections


def run_comparison(pytorch_dets: list, onnx_dets: list, iou_threshold: float = 0.5) -> None:
    """Compare PyTorch and ONNX detections."""
    comparator = DetectionComparator(iou_threshold=iou_threshold)
    result = comparator.compare(pytorch_dets, onnx_dets)
    comparator.print_report(result, first_name="PyTorch", second_name="ONNX")


def run_full_pipeline(args, config: Config) -> None:
    """Run the complete inference pipeline."""
    logger = get_logger()
    
    logger.info("Starting full pipeline")
    
    # 1. PyTorch inference
    pytorch_detections = run_pytorch_inference(args, config)
    
    # 2. ONNX conversion
    onnx_path = run_onnx_conversion(args, config)
    
    # 3. ONNX inference
    onnx_detections = run_onnx_inference(args, config, onnx_path)
    
    # 4. Comparison
    run_comparison(pytorch_detections, onnx_detections)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output images saved to: {config.paths.output_dir}")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="YOLO11 Inference Pipeline - PyTorch and ONNX inference with comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Full pipeline command
    full_parser = subparsers.add_parser("run", help="Run full pipeline (PyTorch + ONNX + comparison)")
    full_parser.add_argument("--image", "-i", type=Path, help="Path to input image")
    full_parser.add_argument("--model", "-m", type=Path, help="Path to PyTorch model")
    
    # PyTorch inference only
    pytorch_parser = subparsers.add_parser("pytorch", help="Run PyTorch inference only")
    pytorch_parser.add_argument("--image", "-i", type=Path, required=True, help="Path to input image")
    pytorch_parser.add_argument("--model", "-m", type=Path, help="Path to PyTorch model")
    
    # Convert to ONNX
    convert_parser = subparsers.add_parser("convert", help="Convert PyTorch model to ONNX")
    convert_parser.add_argument("--model", "-m", type=Path, help="Path to PyTorch model")
    convert_parser.add_argument("--output", "-o", type=Path, help="Output path for ONNX model")
    
    # ONNX inference only
    onnx_parser = subparsers.add_parser("onnx", help="Run ONNX inference only")
    onnx_parser.add_argument("--image", "-i", type=Path, required=True, help="Path to input image")
    onnx_parser.add_argument("--model", "-m", type=Path, help="Path to ONNX model")
    
    return parser


def main(argv=None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logger()
    logger = get_logger()
    
    # Load configuration
    config = Config.from_defaults()
    config.ensure_directories()
    
    try:
        if args.command == "run":
            if not args.image:
                args.image = config.paths.images_dir / "image-2.png"
            run_full_pipeline(args, config)
            
        elif args.command == "pytorch":
            run_pytorch_inference(args, config)
            
        elif args.command == "convert":
            if args.output:
                config.model.onnx_model = args.output
            run_onnx_conversion(args, config)
            
        elif args.command == "onnx":
            onnx_path = args.model or config.model.onnx_model
            if not Path(onnx_path).exists():
                logger.error(f"ONNX model not found: {onnx_path}. Run 'convert' first.")
                return 1
            run_onnx_inference(args, config, onnx_path)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
