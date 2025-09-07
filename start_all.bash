#!/bin/bash
source /root/venv/bin/activate

echo "Starting TianWan AI Detection Services..."

# Debug: Show which python is being used
echo "Using python: $(which python)"
echo "Python version: $(python --version)"

# Debug: Check YOLOX installation
echo "Checking YOLOX installation:"
python -c "import yolox; print('YOLOX version:', yolox.__version__)" || echo "YOLOX not installed"
python -c "import yolox.data; print('YOLOX data module loaded successfully')" || echo "YOLOX data module not found"

# Verify critical dependencies
python -c "import numpy, cv2, torch, requests, flask; print('All dependencies verified')" || {
    echo "ERROR: Dependencies missing!"
    exit 1
}

# Install gunicorn if not present
pip show gunicorn > /dev/null 2>&1 || pip install gunicorn

# Start unified application with gunicorn
echo "Starting unified TianWan AI service on port 8080 with gunicorn..."
NPROC=${NPROC:-1}
gunicorn -w $NPROC --threads 1 -b 0.0.0.0:8080 'main:create_app()'
