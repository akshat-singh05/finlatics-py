"""
Finlatics-Py: web app and REST API for Advertising Analytics & Predictions

Run:  python app.py           (web app at http://127.0.0.1:5000)
Endpoints:
    GET  /                        - Web app (overview, breakdowns, segments,
                                    20 questions, predictor, model, data)
    GET  /health                  - Health check (data / model availability)
    GET  /api/info                - Dataset summary, valid filters, model metadata
    GET  /metrics                 - Overall KPIs
         ?campaign_number=camp 1&user_engagement=High&banner=240 x 400
         &placement=ghi&start=2020-04-01&end=2020-04-30
    GET  /metrics/<dimension>     - KPIs per campaign_number, user_engagement,
                                    banner, placement, month, weekday, date
    GET  /pivot?rows=banner&cols=placement
                                  - KPIs for every pair of two dimensions
    GET  /data                    - Row-level data, paginated (page, page_size,
                                    sort, order) and filtered like /metrics
    GET  /data.csv                - Filtered rows as a CSV download
    GET  /insights                - EDA findings (reports/eda_findings.json)
    GET  /model                   - Model vs baselines (reports/model_evaluation.json)
    GET  /report/<eda|model>      - Written reports as Markdown
    POST /predict                 - Revenue / margin for planned buys (object or list)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from api import create_app
from api.routes.health import API_VERSION
from utils.logger import get_logger

logger = get_logger("api")


app = create_app()


if __name__ == "__main__":
    # Localhost by default; set API_HOST=0.0.0.0 to expose on the network
    host = os.environ.get("API_HOST", "127.0.0.1")
    port = int(os.environ.get("API_PORT", "5000"))
    logger.info("Starting Finlatics-Py API server on %s:%d", host, port)
    print(__doc__)
    print(f"  Finlatics-Py API v{API_VERSION} -> http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
