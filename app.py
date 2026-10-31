"""
Finlatics-Py: REST API for Advertising Analytics & Predictions

Run:  python app.py
Endpoints:
    GET  /health          - Health check
    GET  /api/info        - Model metadata & valid categories
    GET  /metrics          - Overall KPI metrics
    GET  /metrics?campaign=X&channel=Y&device=Z
    GET  /metrics/campaign - Per-campaign breakdown
    GET  /metrics/channel  - Per-channel breakdown
    GET  /metrics/device   - Per-device breakdown
    POST /predict          - Predict profitability from JSON input
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from api import create_app
from utils.logger import get_logger

logger = get_logger("api")


app = create_app()


if __name__ == "__main__":
    logger.info("Starting Finlatics-Py API server")
    print("=" * 50)
    print("  Finlatics-Py API Server v2.0.0")
    print("=" * 50)
    print(f"  Data:   {PROJECT_ROOT / 'data'}")
    print(f"  Models: {PROJECT_ROOT / 'models' / 'saved'}")
    print("  Endpoints:")
    print("    GET  /health")
    print("    GET  /api/info")
    print("    GET  /metrics")
    print("    GET  /metrics?campaign=X&channel=Y&device=Z")
    print("    GET  /metrics/campaign")
    print("    GET  /metrics/channel")
    print("    GET  /metrics/device")
    print("    POST /predict")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
