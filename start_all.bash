#!/bin/bash
source /root/venv/bin/activate

echo "Starting TianWan AI Detection Services..."

# Debug: Show which python is being used
echo "Using python: $(which python)"
echo "Python version: $(python --version)"

# Debug: Check YOLOX installation:
echo "Checking YOLOX installation:"
python -c "import yolox; print('YOLOX version:', yolox.__version__); print('YOLOX location:', yolox.__file__)" || echo "YOLOX not installed"
python -c "import sys; sys.path.insert(0, '/root/YOLO-main-fire'); import yolox.data; print('YOLOX data module loaded successfully')" || echo "YOLOX data module not found"
echo "Checking installed packages:"
pip list | grep -i yolo
echo "Checking YOLO directory structure:"
ls -la /root/YOLO-main-fire/yolox/
ls -la /root/YOLO-main-fire/yolox/data/ || echo "Data directory not found"

# Verify critical dependencies
python -c "import numpy, cv2, torch, requests, flask; print('All dependencies verified')" || {
    echo "ERROR: Dependencies missing!"
    exit 1
}

# Install gunicorn if not present
pip show gunicorn > /dev/null 2>&1 || pip install gunicorn

# Run detailed YOLOX test
echo "Running detailed YOLOX test:"
python test_yolox.py

# Start unified application with gunicorn
echo "Starting unified TianWan AI service on port 8080 with gunicorn..."
NPROC=${NPROC:-1}
gunicorn -w $NPROC --threads 1 -b 0.0.0.0:8080 'main:create_app()'
