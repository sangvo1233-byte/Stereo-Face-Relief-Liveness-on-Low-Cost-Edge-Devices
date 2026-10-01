# Face Attendance System - Docker image target cho Raspberry Pi 4 (ARM64 / aarch64).
# Build tren Pi target:   docker build -t face-attendance .
# Hoac build da kien truc tu PC:  docker buildx build --platform linux/arm64 -t face-attendance .

FROM python:3.11-slim-bookworm

# Goi he thong can cho OpenCV + MediaPipe (libGL, glib, v.v.).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_ENV=production

WORKDIR /app

# Cai dependency truoc (tan dung cache layer khi code doi ma deps khong doi).
# Dung requirements-pi.txt: phien ban DA GHIM cho Pi target.
# Can log build/wheel that tren Pi truoc khi ghi la da verify.
# --prefer-binary tranh roi vao build tu source.
COPY requirements-pi.txt ./
RUN pip install --upgrade pip \
    && pip install --prefer-binary -r requirements-pi.txt

# Copy toan bo source (model/db/logs bi .dockerignore loai ra).
COPY . .

# Thu muc runtime ghi duoc (model tai ve, sqlite, logs).
RUN mkdir -p models database logs \
    && chmod +x docker/entrypoint.sh

EXPOSE 8000

# Health check goi endpoint /health cua app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
