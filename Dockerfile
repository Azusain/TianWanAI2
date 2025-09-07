# Multi-stage build Dockerfile for TianWan AI Detection Services
# Stage 1: Python dependencies builder
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS python-builder

ENV WORKDIR=/app
WORKDIR ${WORKDIR}

# Install Python and create virtual environment
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

# Copy all requirements files
COPY requirements.txt ./
COPY YOLO-main-fire/requirements.txt ./fire-requirements.txt
COPY YOLO-main-helmet/requirements.txt ./helmet-requirements.txt
COPY YOLO-main-safetybelt/requirements.txt ./safetybelt-requirements.txt

# Install Python dependencies using virtual environment activation
RUN . venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r fire-requirements.txt && \
    pip install --no-cache-dir -r helmet-requirements.txt && \
    pip install --no-cache-dir -r safetybelt-requirements.txt && \
    # Clean up unnecessary files in venv
    find venv -name "*.pyc" -delete && \
    find venv -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find venv/lib -name "*.pyi" -delete && \
    find venv/lib -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find venv/lib -name "test" -type d -exec rm -rf {} + 2>/dev/null || true

# Stage 2: Final runtime image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS runtime

ENV WORKDIR=/app
ENV PATH="${WORKDIR}/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}

# Install only runtime dependencies
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy virtual environment from builder stage
COPY --from=python-builder ${WORKDIR}/venv ${WORKDIR}/venv

# Copy application code
COPY YOLO-main-fire/ ./YOLO-main-fire/
COPY YOLO-main-helmet/ ./YOLO-main-helmet/
COPY YOLO-main-safetybelt/ ./YOLO-main-safetybelt/
COPY models/ ./models/
COPY fire_service.py ./
COPY helmet_service.py ./
COPY safetybelt_service.py ./
COPY gateway.py ./
COPY start_all.bash ./

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 || true

# Make startup script executable
RUN chmod +x start_all.bash

# Expose gateway port
EXPOSE 8080

CMD ["bash", "start_all.bash"]
