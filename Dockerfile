# One image for the Spark master/workers and all Python services, so the driver and the
# workers always run the same Python and library versions.
FROM apache/spark:4.0.1

USER root
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY configs ./configs
COPY scripts ./scripts
COPY experiments ./experiments

ENV PYTHONUNBUFFERED=1 \
    PYSPARK_PYTHON=python3 \
    TAXIDRIFT_CONFIG=/app/configs/docker.yaml
