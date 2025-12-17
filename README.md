# YOLO11 Inference Pipeline

A production-ready pipeline for YOLO11 model inference with PyTorch and ONNX support, including model conversion, detection comparison, and a modern web UI.

## Features

- **PyTorch Inference**: Run object detection using the native PyTorch model
- **ONNX Conversion**: Convert PyTorch models to ONNX format with validation
- **ONNX Inference**: Run inference using the optimized ONNX runtime
- **IoU Comparison**: Compare detections between PyTorch and ONNX models
- **Web UI**: Modern, responsive interface with drag-and-drop image upload
- **Threshold Controls**: Adjustable confidence and IoU thresholds via sliders
- **Chart Visualization**: Interactive IoU and confidence comparison charts
- **Image Lightbox**: Click-to-enlarge result images for detailed inspection

## Project Structure

```
elunic-yolo/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   ├── cli.py             # Command-line interface
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py         # Flask application factory
│   │   ├── routes.py      # API endpoints
│   │   └── views.py       # UI routes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── inference.py   # YOLO inference handler
│   │   └── converter.py   # ONNX conversion
│   ├── templates/
│   │   ├── base.html      # Base template
│   │   └── index.html     # Main UI page
│   └── utils/
│       ├── __init__.py
│       └── comparison.py  # Detection comparison utilities
├── data/
│   ├── checkpoints/       # Model files (.pt, .onnx)
│   └── images/            # Input images
├── output/                # Generated outputs
├── docs/
│   └── task.txt           # Task description
├── run.py                 # CLI entry point
├── server.py              # Flask server entry point
├── requirements.txt
├── .env.example           # Environment variables template
└── README.md
```

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Pipeline
Executes PyTorch inference, ONNX conversion, ONNX inference, and comparison:
```bash
python run.py run
```

With custom image:
```bash
python run.py run --image path/to/image.png
```

### PyTorch Inference Only
```bash
python run.py pytorch --image data/images/image-2.png
```

### Convert to ONNX
```bash
python run.py convert
```

### ONNX Inference Only
```bash
python run.py onnx --image data/images/image-2.png
```

## Output

- `output/pytorch_result.png` - Annotated image from PyTorch model
- `output/onnx_result.png` - Annotated image from ONNX model
- Console logs with detection details and IoU comparison statistics

## Configuration

Default configuration can be modified in `src/config.py`:
- Model paths
- Input image size (default: 640)
- Confidence threshold (default: 0.25)
- IoU threshold for NMS (default: 0.45)

## Web UI

Start the Flask server and access the web interface:

```bash
python server.py
```

Open `http://localhost:5000` in your browser.

### UI Features

- **Model Status**: View availability of PyTorch and ONNX models
- **Image Upload**: Drag-and-drop or click to upload images
- **Threshold Sliders**:
  - Confidence Threshold (0.05 - 0.95)
  - NMS IoU Threshold (0.1 - 0.9)
  - Comparison IoU Threshold (0.1 - 0.9)
- **Inference Buttons**: Run PyTorch, ONNX, or comparison inference
- **Results Display**: View annotated images with detection counts
- **Chart Analysis** (Comparison mode):
  - IoU bar chart with threshold line
  - Confidence comparison chart (PyTorch vs ONNX)
- **Image Lightbox**: Click any result image to view fullscreen

## REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/models/status` | Check model availability |
| POST | `/api/predict/pytorch` | Run PyTorch inference |
| POST | `/api/predict/onnx` | Run ONNX inference |
| POST | `/api/convert` | Convert PyTorch to ONNX |
| POST | `/api/compare` | Compare both models |
| GET | `/api/output/<filename>` | Get output image |

### Request Parameters

All inference endpoints accept the following form parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | file | required | Image file to process |
| `conf_threshold` | float | 0.25 | Confidence threshold |
| `iou_threshold` | float | 0.45 | NMS IoU threshold |
| `comparison_iou_threshold` | float | 0.5 | IoU threshold for matching (compare only) |

### Example Usage

**PyTorch Inference with custom thresholds:**
```bash
curl -X POST -F "image=@path/to/image.png" \
     -F "conf_threshold=0.5" \
     -F "iou_threshold=0.4" \
     http://localhost:5000/api/predict/pytorch
```

**ONNX Conversion:**
```bash
curl -X POST http://localhost:5000/api/convert
```

**Compare Models:**
```bash
curl -X POST -F "image=@path/to/image.png" \
     -F "comparison_iou_threshold=0.6" \
     http://localhost:5000/api/compare
```

### Response Format

**Single Model Inference:**
```json
{
  "model_type": "pytorch",
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "confidence": 0.95,
      "class_id": 0,
      "class_name": "person"
    }
  ],
  "count": 1,
  "output_image": "pytorch_image.png"
}
```

**Comparison Response:**
```json
{
  "pytorch": { "count": 5, "output_image": "pytorch_image.png" },
  "onnx": { "count": 5, "output_image": "onnx_image.png" },
  "comparison": {
    "matched_count": 4,
    "mean_iou": 0.92,
    "min_iou": 0.85,
    "max_iou": 0.98,
    "pytorch_only": 1,
    "onnx_only": 1,
    "iou_threshold_used": 0.5
  },
  "chart_data": [
    {
      "index": 1,
      "class_name": "person",
      "iou": 0.95,
      "pytorch_conf": 0.92,
      "onnx_conf": 0.91
    }
  ]
}
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | 1 | Enable debug mode |
| `PORT` | 5000 | Server port |
