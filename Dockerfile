# Multi-stage build Dockerfile - optimized for size and build efficiency
# Stage 1: Python dependencies builder
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS python-builder

ENV WORKDIR=/root
WORKDIR ${WORKDIR}

# Install Python and build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        gcc \
        g++ \
        && \
    python3.11 -m venv venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt YOLO-main-safetybelt/requirements.txt ./temp/
RUN . venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir flask gunicorn loguru ultralytics opencv-python numpy && \
    # Clean up unnecessary files in venv
    find venv -name "*.pyc" -delete && \
    find venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find venv/lib -name "*.pyi" -delete && \
    find venv/lib -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find venv/lib -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: YOLOX installation
FROM python-builder AS yolox-builder

# Copy YOLO project and install
COPY YOLO-main-safetybelt ./YOLO-main-safetybelt
RUN . venv/bin/activate && \
    cd YOLO-main-safetybelt && \
    pip install --no-cache-dir -e . && \
    cd .. && \
    # Clean up after YOLOX installation
    find venv -name "*.pyc" -delete && \
    find venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 3: Final runtime image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime

ENV WORKDIR=/root
ENV PATH="${WORKDIR}/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}

# Install only runtime dependencies
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy virtual environment from builder stages
COPY --from=yolox-builder ${WORKDIR}/venv ${WORKDIR}/venv
COPY --from=yolox-builder ${WORKDIR}/YOLO-main-safetybelt ${WORKDIR}/YOLO-main-safetybelt

# Copy application code and models
COPY models/ ./models/
COPY safetybelt_service.py ./

# Create symlink for python if needed
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 || true

# Expose port for safetybelt service
EXPOSE 8080

# Run safety belt service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "safetybelt_service:app"]
