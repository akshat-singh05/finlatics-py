# Reproducible environment for the full pipeline (analysis + ML).
#
# Build:  docker build -t finlatics-py .
# Run:    docker run --rm -v "${PWD}/reports:/app/reports" -v "${PWD}/output:/app/output" \
#                    -v "${PWD}/models/saved:/app/models/saved" finlatics-py
# Tests:  docker run --rm finlatics-py pytest
#
# The mounts write charts, reports and the trained model back to the host.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    N_JOBS=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
