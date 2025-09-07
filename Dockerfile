# Single-stage build Dockerfile for TianWan AI Detection Services
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV WORKDIR=/root
ENV PATH="${WORKDIR}/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}

# Install system dependencies and Python
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
    python3.11 -m venv venv --system-site-packages && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files first
COPY requirements.txt ./
COPY YOLO-main-fire/requirements.txt ./fire-requirements.txt
COPY YOLO-main-helmet/requirements.txt ./helmet-requirements.txt
COPY YOLO-main-safetybelt/requirements.txt ./safetybelt-requirements.txt

# Install Python dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt && \
    pip install -r fire-requirements.txt && \
    pip install -r helmet-requirements.txt && \
    pip install -r safetybelt-requirements.txt

# Install YOLOX packages after copying the code
COPY YOLO-main-fire/ ./YOLO-main-fire/
COPY YOLO-main-helmet/ ./YOLO-main-helmet/
COPY YOLO-main-safetybelt/ ./YOLO-main-safetybelt/

# Install only one YOLOX package to avoid conflicts (they're all similar)
RUN . venv/bin/activate && cd YOLO-main-fire && pip install -e . && cd .. && pip list | grep yolox

# Copy remaining application code
COPY models/ ./models/
COPY fire_service.py ./
COPY helmet_service.py ./
COPY safetybelt_service.py ./
COPY gateway.py ./
COPY main.py ./
COPY api.py ./
COPY test_yolox.py ./
COPY start_all.bash ./

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 || true

# Make startup script executable
RUN chmod +x start_all.bash

# 设置 PATH 在最后
ENV PATH="${WORKDIR}/venv/bin:$PATH"

# Expose gateway port
EXPOSE 8080

CMD ["bash", "start_all.bash"]
