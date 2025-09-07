#!/bin/bash
source /root/venv/bin/activate

echo "Starting TianWan AI Detection Microservices..."

# Verify dependencies
python -c "import numpy, cv2, torch, flask, yolox; print('All dependencies verified')" || {
    echo "ERROR: Dependencies missing!"
    exit 1
}

# Start fire detection service
echo "Starting Fire Detection Service on port 8901..."
cd /root && gunicorn --bind 0.0.0.0:8901 --workers 1 --timeout 300 fire_service:app &
FIRE_PID=$!

# Start helmet detection service  
echo "Starting Helmet Detection Service on port 8902..."
cd /root && gunicorn --bind 0.0.0.0:8902 --workers 1 --timeout 300 helmet_service:app &
HELMET_PID=$!

# Start safetybelt detection service
echo "Starting Safetybelt Detection Service on port 8903..."
cd /root && gunicorn --bind 0.0.0.0:8903 --workers 1 --timeout 300 safetybelt_service:app &
SAFETYBELT_PID=$!

# Wait for services to start
sleep 10

# Start API Gateway (use gateway.py with Flask dev server for now)
echo "Starting API Gateway on port 8080..."
cd /root && python gateway.py &
GATEWAY_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down services..."
    kill $FIRE_PID $HELMET_PID $SAFETYBELT_PID $GATEWAY_PID 2>/dev/null
    exit 0
}

# Trap signals
trap shutdown SIGINT SIGTERM

# Keep the script running
while true; do
    # Check if processes are still running
    if ! kill -0 $GATEWAY_PID 2>/dev/null; then
        echo "Gateway stopped, restarting all services..."
        shutdown
        exec "$0"
    fi
    sleep 10
done
