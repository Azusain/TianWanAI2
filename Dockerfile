# TianWan AI Detection Microservices Docker Image
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

# Copy YOLO projects to their own directories
COPY YOLO-main-fire ./YOLO-main-fire
COPY YOLO-main-helmet ./YOLO-main-helmet
COPY YOLO-main-safetybelt ./YOLO-main-safetybelt

# Install YOLOX in editable mode for each project
RUN cd YOLO-main-fire && pip install -e . && \
    cd ../YOLO-main-helmet && pip install -e . && \
    cd ../YOLO-main-safetybelt && pip install -e .

# Copy models and services
COPY models/ ./models/
COPY fire_service.py helmet_service.py safetybelt_service.py gateway.py ./
COPY run.bash ./

# Make startup script executable
RUN chmod +x run.bash

# Use microservices architecture
CMD ["bash", "run.bash"]
