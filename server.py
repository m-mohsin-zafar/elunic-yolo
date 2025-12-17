#!/usr/bin/env python
"""Flask server entry point."""

import os
from src.api import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    
    print(f"Starting YOLO Inference API on http://localhost:{port}")
    print("Available endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/models/status   - Check model availability")
    print("  POST /api/predict/pytorch - PyTorch inference")
    print("  POST /api/predict/onnx    - ONNX inference")
    print("  POST /api/convert         - Convert to ONNX")
    print("  POST /api/compare         - Compare both models")
    print("  GET  /api/output/<file>   - Get output image")
    
    app.run(host="0.0.0.0", port=port, debug=debug)
