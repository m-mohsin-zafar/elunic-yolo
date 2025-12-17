"""Flask application factory."""

import os
from pathlib import Path

from flask import Flask

from ..config import Config
from ..logger import setup_logger


def create_app(config: Config = None) -> Flask:
    """
    Application factory for Flask app.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured Flask application
    """
    # Set template folder path
    template_dir = Path(__file__).parent.parent / "templates"
    
    app = Flask(__name__, template_folder=str(template_dir))
    
    # Setup logging
    setup_logger()
    
    # Load configuration
    if config is None:
        config = Config.from_defaults()
    
    config.ensure_directories()
    
    # Store config in app
    app.config["YOLO_CONFIG"] = config
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload
    app.config["UPLOAD_FOLDER"] = str(config.paths.images_dir)
    
    # Register blueprints
    from .routes import api_bp
    from .views import views_bp
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(views_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    return app


def register_error_handlers(app: Flask) -> None:
    """Register error handlers for the application."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {"error": "Bad request", "message": str(error.description)}, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found", "message": str(error.description)}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error", "message": "An unexpected error occurred"}, 500
