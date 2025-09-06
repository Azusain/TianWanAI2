#!/usr/bin/env python3
import requests
from flask import Flask, request, jsonify
from loguru import logger
from uuid import uuid4

# Microservice endpoints - all use /predict internally
SERVICES = {
    "fire": "http://localhost:8901/predict",
    "helmet": "http://localhost:8902/predict",
    "safetybelt": "http://localhost:8903/predict"
}

VERSION_API = '0.0.1'

def forward_request(service_name, request_data):
    """Forward request to microservice"""
    try:
        if service_name not in SERVICES:
            return None, -5
        
        # Forward request to microservice
        response = requests.post(
            SERVICES[service_name], 
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json(), 0
        else:
            logger.error(f"service {service_name} returned status {response.status_code}")
            return None, -5
            
    except requests.exceptions.Timeout:
        logger.error(f"timeout waiting for {service_name} service")
        return None, -6
    except Exception as e:
        logger.error(f"error forwarding request to {service_name}: {e}")
        return None, -5

# Create Flask app
app = Flask(__name__)

@app.route('/fire', methods=['POST'])
def fire_detect():
    """Fire detection endpoint - forwards to fire service"""
    try:
        request_data = request.get_json()
        result, errno = forward_request("fire", request_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({
                "log_id": str(uuid4()),
                "errno": errno,
                "err_msg": "service error",
                "api_version": VERSION_API,
                "model_name": "fire",
                "results": []
            })
            
    except Exception as e:
        logger.error(f"gateway error in fire endpoint: {e}")
        return jsonify({
            "log_id": str(uuid4()),
            "errno": -5,
            "err_msg": "gateway error",
            "api_version": VERSION_API,
            "model_name": "fire",
            "results": []
        })

@app.route('/helmet', methods=['POST'])
def helmet_detect():
    """Helmet safety detection endpoint - forwards to helmet service"""
    try:
        request_data = request.get_json()
        result, errno = forward_request("helmet", request_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({
                "log_id": str(uuid4()),
                "errno": errno,
                "err_msg": "service error",
                "api_version": VERSION_API,
                "model_name": "helmet_safety",
                "results": []
            })
            
    except Exception as e:
        logger.error(f"gateway error in helmet endpoint: {e}")
        return jsonify({
            "log_id": str(uuid4()),
            "errno": -5,
            "err_msg": "gateway error",
            "api_version": VERSION_API,
            "model_name": "helmet_safety",
            "results": []
        })

@app.route('/safetybelt', methods=['POST'])
def safetybelt_detect():
    """Safety belt detection endpoint - forwards to safetybelt service"""
    try:
        request_data = request.get_json()
        result, errno = forward_request("safetybelt", request_data)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({
                "log_id": str(uuid4()),
                "errno": errno,
                "err_msg": "service error",
                "api_version": VERSION_API,
                "model_name": "safetybelt_safety",
                "results": []
            })
            
    except Exception as e:
        logger.error(f"gateway error in safetybelt endpoint: {e}")
        return jsonify({
            "log_id": str(uuid4()),
            "errno": -5,
            "err_msg": "gateway error",
            "api_version": VERSION_API,
            "model_name": "safetybelt_safety",
            "results": []
        })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        "available_models": [
            {
                "name": "fire",
                "endpoint": "/fire",
                "description": "fire detection model"
            },
            {
                "name": "helmet", 
                "endpoint": "/helmet",
                "description": "helmet safety violation detection"
            },
            {
                "name": "safetybelt",
                "endpoint": "/safetybelt", 
                "description": "safety belt violation detection"
            }
        ]
    })

if __name__ == '__main__':
    logger.info("starting API gateway on port 8080...")
    logger.info("routing table:")
    logger.info("  GET  /models      -> list available models")
    logger.info("  POST /fire        -> http://localhost:8901/predict")
    logger.info("  POST /helmet      -> http://localhost:8902/predict")
    logger.info("  POST /safetybelt  -> http://localhost:8903/predict")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
