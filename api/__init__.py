"""
Finlatics-Py API package.

Provides a Flask application factory with Blueprint-based route organization.

Usage:
    from api import create_app
    app = create_app()
    app.run()
"""

import time
from pathlib import Path
from flask import Flask, request, jsonify, g

from utils.logger import get_logger, LogContext

logger = get_logger("api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)

    try:
        from flask_cors import CORS
        CORS(app)
        logger.info("CORS enabled")
    except ImportError:
        logger.warning("flask-cors not installed — CORS disabled")

    # Register Blueprints
    from api.routes.health import health_bp, API_VERSION
    from api.routes.metrics import metrics_bp
    from api.routes.predict import predict_bp
    from api.routes.ui import ui_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(ui_bp)

    # Request lifecycle hooks
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.request_id = request.headers.get("X-Request-ID", "-")

    @app.after_request
    def after_request(response):
        elapsed = (time.time() - g.start_time) * 1000
        with LogContext(request_id=g.request_id):
            logger.info(
                "%s %s %s %.1fms",
                request.method, request.path, response.status_code, elapsed,
            )
        response.headers["X-API-Version"] = API_VERSION
        response.headers["X-Request-ID"] = g.request_id
        return response

    # Error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({"error": "Bad request", "message": str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

    logger.info("Flask app created with %d blueprints", len(app.blueprints))
    return app
