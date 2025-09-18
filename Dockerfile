# TianWan2 Safety Belt Detection Docker Image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV WORKDIR=/root
ENV PATH="${WORKDIR}/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}

# Install system dependencies with all Python standard library components
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        gcc \
        g++ \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && \
    python3.11 -m venv --system-site-packages venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt && \
    pip install requests gunicorn

# Copy YOLO safety belt project
COPY YOLO-main-safetybelt ./YOLO-main-safetybelt

# Install YOLOX in editable mode
RUN cd YOLO-main-safetybelt && pip install -e .

# Copy models and service
COPY models/ ./models/
COPY safetybelt_service.py ./

# Expose port for safetybelt service
EXPOSE 8080

# Run safety belt service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "safetybelt_service:app"]
