import binascii
import os
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from loguru import logger
from uuid import uuid4
from api import ServiceStatus, YOLOXDetectionService, PersonDetectionService, SafetyDetectionService

# Global model instances (initialized at startup)
g_fire_service = None
g_helmet_service = None  
g_safetybelt_service = None
g_person_detector = None

def validate_img_format():
    """Validate and decode base64 image from request"""
    req = None
    try:
        req = request.json              
    except Exception:         
        return None, ServiceStatus.INVALID_CONTENT_TYPE.value
        
    # check if the json data contains 'image' key
    if not req or not req.get('image'):         
        return None, ServiceStatus.MISSING_IMAGE_DATA.value
    
    # check whether the image data has a valid format
    try:
        if type(req['image']) is not str:
            raise binascii.Error
        bin_data = base64.b64decode(req['image'])     
        np_arr = np.frombuffer(bin_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   
        if img is None:
            raise binascii.Error
        return img, None
    except binascii.Error: 
        return None, ServiceStatus.INVALID_IMAGE_FORMAT.value

def initialize_models():
    """Initialize all models at startup"""
    global g_fire_service, g_helmet_service, g_safetybelt_service, g_person_detector
    
    logger.info("starting model initialization...")
    
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize person detection model first
        logger.info("loading person detection model...")
        person_model_path = os.path.join(base_dir, "models", "yolo11s.pt")
        g_person_detector = PersonDetectionService(
            model_path=person_model_path
        )
        
        # Initialize fire detection model (standalone)
        logger.info("loading fire detection model...")
        fire_model_dir = os.path.join(base_dir, "YOLO-main-fire")
        g_fire_service = YOLOXDetectionService(
            model_name="fire",
            model_dir=fire_model_dir,
            exp_file="exps/example/yolox_voc/yolox_voc_s.py", 
            ckpt_file="weights/best.pth"
        )
        
        # Initialize helmet safety detection model (with person detection)
        logger.info("loading helmet safety detection model...")
        helmet_model_dir = os.path.join(base_dir, "YOLO-main-helmet")
        g_helmet_service = SafetyDetectionService(
            model_name="helmet_safety",
            model_dir=helmet_model_dir,
            exp_file="exps/example/yolox_voc/yolox_voc_s.py",
            ckpt_file="weights/best.pth",
            person_detector=g_person_detector
        )
        
        # Initialize safety belt detection model (with person detection)
        logger.info("loading safety belt detection model...")
        safetybelt_model_dir = os.path.join(base_dir, "YOLO-main-safetybelt")
        g_safetybelt_service = SafetyDetectionService(
            model_name="safetybelt_safety", 
            model_dir=safetybelt_model_dir,
            exp_file="exps/example/yolox_voc/yolox_voc_s.py",
            ckpt_file="weights/best_ckpt.pth",
            person_detector=g_person_detector
        )
        
        logger.success("all models initialized successfully!")
        
    except Exception as e:
        logger.error(f"failed to initialize models: {e}")
        raise e

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Initialize all models at startup
    initialize_models()
    logger.info("server is up!")
    
    @app.route('/fire', methods=['POST'])
    def fire_detect():
        """Fire detection endpoint"""
        img, errno = validate_img_format()
        if img is None:
            return g_fire_service.response(errno=errno)
        
        # inference
        results, errno = g_fire_service.predict(img)
        return g_fire_service.response(errno=errno, results=results)
    
    @app.route('/helmet', methods=['POST']) 
    def helmet_detect():
        """Helmet safety detection endpoint - detects persons without helmets"""
        img, errno = validate_img_format()
        if img is None:
            return g_helmet_service.response(errno=errno)
        
        # inference using safety detection (person + helmet)
        results, errno = g_helmet_service.predict_safety(img)
        return g_helmet_service.response(errno=errno, results=results)
    
    @app.route('/safetybelt', methods=['POST'])
    def safetybelt_detect():
        """Safety belt detection endpoint - detects persons without safety belts"""
        img, errno = validate_img_format()
        if img is None:
            return g_safetybelt_service.response(errno=errno)
        
        # inference using safety detection (person + safetybelt)
        results, errno = g_safetybelt_service.predict_safety(img)
        return g_safetybelt_service.response(errno=errno, results=results)
    
    @app.route('/models', methods=['GET'])
    def list_models():
        """List available models"""
        return {
            "available_models": [
                {
                    "name": "fire",
                    "endpoint": "/fire",
                    "description": "fire detection model"
                },
                {
                    "name": "helmet", 
                    "endpoint": "/helmet",
                    "description": "helmet safety violation detection - detects persons without helmets"
                },
                {
                    "name": "safetybelt",
                    "endpoint": "/safetybelt", 
                    "description": "safety belt violation detection - detects persons without safety belts"
                }
            ]
        }
    
    return app

if __name__ == "__main__":
    # Setup logging
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "runtime.log")
    logger.add(
        log_path,
        rotation="2 GB",
        retention="7 days"
    )
    
    app = create_app()
    app.run(port=8091, debug=False, host='0.0.0.0')
