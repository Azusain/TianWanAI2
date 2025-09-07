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
cd /root/YOLO-main-fire && python /root/fire_service.py &
FIRE_PID=$!

# Start helmet detection service  
echo "Starting Helmet Detection Service on port 8902..."
cd /root/YOLO-main-helmet && python /root/helmet_service.py &
HELMET_PID=$!

# Start safetybelt detection service
echo "Starting Safetybelt Detection Service on port 8903..."
cd /root/YOLO-main-safetybelt && python /root/safetybelt_service.py &
SAFETYBELT_PID=$!

# Wait for services to start
sleep 5

# Start unified gateway with gunicorn
echo "Starting API Gateway on port 8080..."
cd /root && gunicorn -w 1 --threads 1 -b 0.0.0.0:8080 'main:create_app()' &
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
