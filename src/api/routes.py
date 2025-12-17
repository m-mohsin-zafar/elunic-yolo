"""API routes for YOLO inference."""

import os
import uuid
from pathlib import Path

from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

from ..models import YOLOInference, ONNXConverter
from ..utils import DetectionComparator
from ..logger import get_logger

api_bp = Blueprint("api", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_config():
    """Get configuration from app context."""
    return current_app.config["YOLO_CONFIG"]


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "yolo-inference-api"})


@api_bp.route("/predict/pytorch", methods=["POST"])
def predict_pytorch():
    """
    Run PyTorch model inference on uploaded image.
    
    Request: multipart/form-data with 'image' file
    Response: JSON with detections
    """
    logger = get_logger()
    config = get_config()
    
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = Path(config.paths.images_dir) / filename
    file.save(str(filepath))
    
    # Get threshold parameters from form data
    conf_threshold = float(request.form.get('conf_threshold', config.model.confidence_threshold))
    iou_threshold = float(request.form.get('iou_threshold', config.model.iou_threshold))
    
    try:
        # Run inference
        model = YOLOInference(config.model.pytorch_model)
        output_path = config.paths.output_dir / f"pytorch_{filename}"
        
        detections = model.predict_and_visualize(
            image_path=filepath,
            output_path=output_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        
        return jsonify({
            "model_type": "pytorch",
            "detections": [d.to_dict() for d in detections],
            "count": len(detections),
            "output_image": str(output_path.name),
        })
        
    except Exception as e:
        logger.exception(f"PyTorch inference failed: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@api_bp.route("/predict/onnx", methods=["POST"])
def predict_onnx():
    """
    Run ONNX model inference on uploaded image.
    
    Request: multipart/form-data with 'image' file
    Response: JSON with detections
    """
    logger = get_logger()
    config = get_config()
    
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    # Check if ONNX model exists
    if not config.model.onnx_model.exists():
        return jsonify({"error": "ONNX model not found. Run /api/convert first."}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = Path(config.paths.images_dir) / filename
    file.save(str(filepath))
    
    # Get threshold parameters from form data
    conf_threshold = float(request.form.get('conf_threshold', config.model.confidence_threshold))
    iou_threshold = float(request.form.get('iou_threshold', config.model.iou_threshold))
    
    try:
        # Run inference
        model = YOLOInference(config.model.onnx_model)
        output_path = config.paths.output_dir / f"onnx_{filename}"
        
        detections = model.predict_and_visualize(
            image_path=filepath,
            output_path=output_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        
        return jsonify({
            "model_type": "onnx",
            "detections": [d.to_dict() for d in detections],
            "count": len(detections),
            "output_image": str(output_path.name),
        })
        
    except Exception as e:
        logger.exception(f"ONNX inference failed: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@api_bp.route("/convert", methods=["POST"])
def convert_to_onnx():
    """
    Convert PyTorch model to ONNX format.
    
    Response: JSON with conversion status
    """
    logger = get_logger()
    config = get_config()
    
    try:
        converter = ONNXConverter(config.model.pytorch_model)
        
        onnx_path = converter.convert(
            output_path=config.model.onnx_model,
            input_size=config.model.input_size,
            simplify=True,
        )
        
        is_valid = converter.validate_onnx(onnx_path)
        
        return jsonify({
            "status": "success",
            "onnx_model": str(onnx_path),
            "valid": is_valid,
        })
        
    except Exception as e:
        logger.exception(f"ONNX conversion failed: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/compare", methods=["POST"])
def compare_models():
    """
    Run both PyTorch and ONNX inference and compare results.
    
    Request: multipart/form-data with 'image' file
    Response: JSON with detections from both models and comparison
    """
    logger = get_logger()
    config = get_config()
    
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400
    
    # Check if ONNX model exists
    if not config.model.onnx_model.exists():
        return jsonify({"error": "ONNX model not found. Run /api/convert first."}), 400
    
    # Save uploaded file
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = Path(config.paths.images_dir) / filename
    file.save(str(filepath))
    
    # Get threshold parameters from form data
    conf_threshold = float(request.form.get('conf_threshold', config.model.confidence_threshold))
    nms_iou_threshold = float(request.form.get('iou_threshold', config.model.iou_threshold))
    comparison_iou_threshold = float(request.form.get('comparison_iou_threshold', 0.5))
    
    try:
        # PyTorch inference
        pytorch_model = YOLOInference(config.model.pytorch_model)
        pytorch_output = config.paths.output_dir / f"pytorch_{filename}"
        pytorch_dets = pytorch_model.predict_and_visualize(
            image_path=filepath,
            output_path=pytorch_output,
            conf_threshold=conf_threshold,
            iou_threshold=nms_iou_threshold,
        )
        
        # ONNX inference
        onnx_model = YOLOInference(config.model.onnx_model)
        onnx_output = config.paths.output_dir / f"onnx_{filename}"
        onnx_dets = onnx_model.predict_and_visualize(
            image_path=filepath,
            output_path=onnx_output,
            conf_threshold=conf_threshold,
            iou_threshold=nms_iou_threshold,
        )
        
        # Compare
        comparator = DetectionComparator(iou_threshold=comparison_iou_threshold)
        result = comparator.compare(pytorch_dets, onnx_dets)
        
        # Build per-detection IoU data for charts
        iou_chart_data = []
        for i, (pt_det, onnx_det, iou) in enumerate(result.matched_pairs):
            iou_chart_data.append({
                "index": i + 1,
                "class_name": pt_det.class_name,
                "iou": round(iou, 4),
                "pytorch_conf": round(pt_det.confidence, 4),
                "onnx_conf": round(onnx_det.confidence, 4),
            })
        
        return jsonify({
            "pytorch": {
                "detections": [d.to_dict() for d in pytorch_dets],
                "count": len(pytorch_dets),
                "output_image": str(pytorch_output.name),
            },
            "onnx": {
                "detections": [d.to_dict() for d in onnx_dets],
                "count": len(onnx_dets),
                "output_image": str(onnx_output.name),
            },
            "comparison": {
                "matched_count": len(result.matched_pairs),
                "pytorch_only": len(result.unmatched_first),
                "onnx_only": len(result.unmatched_second),
                "mean_iou": result.mean_iou,
                "min_iou": result.min_iou,
                "max_iou": result.max_iou,
                "iou_threshold_used": comparison_iou_threshold,
            },
            "chart_data": iou_chart_data,
        })
        
    except Exception as e:
        logger.exception(f"Comparison failed: {e}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up uploaded file
        if filepath.exists():
            filepath.unlink()


@api_bp.route("/output/<filename>", methods=["GET"])
def get_output_image(filename: str):
    """
    Retrieve an output image by filename.
    
    Args:
        filename: Name of the output image file
        
    Returns:
        Image file
    """
    config = get_config()
    filepath = config.paths.output_dir / secure_filename(filename)
    
    if not filepath.exists():
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(str(filepath), mimetype="image/png")


@api_bp.route("/models/status", methods=["GET"])
def models_status():
    """Get status of available models."""
    config = get_config()
    
    return jsonify({
        "pytorch": {
            "path": str(config.model.pytorch_model),
            "exists": config.model.pytorch_model.exists(),
        },
        "onnx": {
            "path": str(config.model.onnx_model),
            "exists": config.model.onnx_model.exists(),
        },
    })
