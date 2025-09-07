#!/bin/bash
source /root/venv/bin/activate

echo "Starting TianWan AI Detection Services..."


# Verify critical dependencies
python -c "import numpy, cv2, torch, requests, flask, yolox; print('All dependencies verified')" || {
    echo "ERROR: Dependencies missing!"
    exit 1
}

# Start unified application with gunicorn
echo "Starting unified TianWan AI service on port 8080..."
NPROC=${NPROC:-1}
gunicorn -w $NPROC --threads 1 -b 0.0.0.0:8080 'main:create_app()'
