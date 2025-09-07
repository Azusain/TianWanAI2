# TianWan AI Detection Services Docker Image
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV WORKDIR=/root
ENV PATH="${WORKDIR}/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR ${WORKDIR}

# Install system dependencies
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        gcc \
        g++ \
        cmake \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        && \
    python3.11 -m venv venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    pip install -r requirements.txt

# Copy YOLO projects completely
COPY YOLO-main-fire ./YOLO-main-fire
COPY YOLO-main-helmet ./YOLO-main-helmet
COPY YOLO-main-safetybelt ./YOLO-main-safetybelt

# Install YOLOX
RUN cd YOLO-main-fire && pip install -e .

# Copy application code
COPY models/ ./models/
COPY main.py api.py start_all.bash ./

# Create symlink for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 || true

# Make startup script executable
RUN chmod +x start_all.bash

# 设置 PATH 在最后
ENV PATH="${WORKDIR}/venv/bin:$PATH"

# Expose gateway port
EXPOSE 8080

CMD ["bash", "start_all.bash"]
