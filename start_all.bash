#!/bin/bash
source /root/venv/bin/activate

echo "Starting TianWan AI Detection Services..."

# Debug: Show which python is being used
echo "Using python: $(which python)"
echo "Python version: $(python --version)"

# Verify critical dependencies
python -c "import numpy, cv2, torch, requests, flask; print('All dependencies verified')" || {
    echo "ERROR: Dependencies missing!"
    exit 1
}

# Start fire detection service in background
echo "Starting Fire Detection Service on port 8901..."
python fire_service.py &
FIRE_PID=$!

# Start helmet safety service in background  
echo "Starting Helmet Safety Service on port 8902..."
python helmet_service.py &
HELMET_PID=$!

# Start safetybelt compliance service in background
echo "Starting Safety Belt Service on port 8903..."
python safetybelt_service.py &
SAFETYBELT_PID=$!

# Wait a moment for services to start
sleep 5

# Start API gateway (foreground)
echo "Starting API Gateway on port 8080..."
python gateway.py &
GATEWAY_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down services..."
    kill $FIRE_PID $HELMET_PID $SAFETYBELT_PID $GATEWAY_PID 2>/dev/null
    exit 0
}

# Trap signals
trap shutdown SIGINT SIGTERM

# Wait for all processes
wait
