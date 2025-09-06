#!/bin/bash
echo "Starting TianWan AI Detection Services..."

# Start fire detection service in background
echo "Starting Fire Detection Service on port 8901..."
/app/venv/bin/python fire_service.py &
FIRE_PID=$!

# Start helmet safety service in background  
echo "Starting Helmet Safety Service on port 8902..."
/app/venv/bin/python helmet_service.py &
HELMET_PID=$!

# Wait a moment for services to start
sleep 5

# Start API gateway (foreground)
echo "Starting API Gateway on port 8080..."
/app/venv/bin/python gateway.py &
GATEWAY_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down services..."
    kill $FIRE_PID $HELMET_PID $GATEWAY_PID 2>/dev/null
    exit 0
}

# Trap signals
trap shutdown SIGINT SIGTERM

# Wait for all processes
wait
