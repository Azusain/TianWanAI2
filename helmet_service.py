#!/usr/bin/env python3
import os
import sys
import base64
import binascii
import numpy as np
import cv2
import torch
import time
import math
import datetime  # TEMPORARY: for debug image naming
from flask import Flask, request, jsonify
from loguru import logger
from uuid import uuid4

# Configure loguru for async logging
logger.remove()  # Remove default handler
logger.add(
    "logs/helmet_service_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    enqueue=True,  # Enable async logging
    backtrace=True,
    diagnose=True,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
)
# Keep console output with proper colors
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
    level="DEBUG",
    enqueue=True
)

# Add helmet model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
helmet_yolo_path = os.path.join(script_dir, "YOLO-main-helmet")
sys.path.insert(0, helmet_yolo_path)

# Import YOLOX modules
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

# TEMPORARY DEBUG FUNCTION - TO BE REMOVED LATER
def save_debug_helmet_image(img, person_idx, helmet_results, has_helmet, violation_score):
    """Save debug image with helmet detection results for visual inspection"""
    try:
        debug_dir = os.path.join(script_dir, "debug_helmet_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        img_copy = img.copy()
        
        # Draw helmet detection results on the image
        if helmet_results:
            for helmet_item in helmet_results:
                if helmet_item["class"] == 0:  # helmet class
                    # Convert normalized coordinates back to pixel coordinates
                    img_h, img_w = img.shape[:2]
                    location = helmet_item["location"]
                    x1 = int((location["left"]) * img_w)
                    y1 = int((location["top"]) * img_h)
                    x2 = int((location["left"] + location["width"]) * img_w)
                    y2 = int((location["top"] + location["height"]) * img_h)
                    
                    # Draw bounding box
                    color = (0, 255, 0) if has_helmet else (0, 0, 255)
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw confidence score
                    conf_text = f"Helmet: {helmet_item['score']:.3f}"
                    cv2.putText(img_copy, conf_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw violation score on top
        violation_text = f"Person {person_idx} - Violation: {violation_score:.3f}"
        has_helmet_text = f"Has Helmet: {has_helmet}"
        cv2.putText(img_copy, violation_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img_copy, has_helmet_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        filename = f"helmet_debug_person{person_idx}_{timestamp}_v{violation_score:.3f}.jpg"
        filepath = os.path.join(debug_dir, filename)
        cv2.imwrite(filepath, img_copy)
        
        logger.info(f"DEBUG: Saved helmet detection image: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save debug helmet image: {e}")

def get_device():
    """Auto-detect device: prefer CUDA, fallback to CPU"""
    if torch.cuda.is_available():
        device = "gpu"  # Use "gpu" for YOLOX compatibility
        logger.info(f"using GPU device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    return device

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=VOC_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("helmet infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

class PersonDetector:
    def __init__(self):
        self.model = None
        self.device = get_device()
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = os.path.join(script_dir, "models", "yolo11s.pt")
            logger.info(f"loading person detection model: {model_path}")
            self.model = YOLO(model_path)
            
            # Move model to appropriate device
            if self.device == "gpu":
                self.model.to("cuda")
                
            logger.info(f"person detection model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"failed to load person detection model: {e}")
            raise e
    
    def detect_persons(self, img, conf_threshold=0.5):
        try:
            results = self.model.predict(img, conf=conf_threshold, classes=[0], verbose=False, device=self.device)
            persons = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        persons.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(conf)
                        })
            
            return persons
        except Exception as e:
            logger.error(f"person detection error: {e}")
            return []

class HelmetPredictor:
    def __init__(self):
        self.model = None
        self.exp = None
        self.predictor = None
        self.device = get_device()
        self._load_model()
    
    def _load_model(self):
        try:
            # Get experiment configuration
            exp_file = os.path.join(helmet_yolo_path, "exps/example/yolox_voc/yolox_voc_s.py")
            self.exp = get_exp(exp_file, None)
            
            # Get model
            self.model = self.exp.get_model()
            logger.info(f"helmet model summary: {get_model_info(self.model, self.exp.test_size)}")
            
            # Set evaluation mode
            self.model.eval()
            
            # Load checkpoint
            ckpt_path = os.path.join(helmet_yolo_path, "weights/best.pth")
            logger.info(f"loading helmet checkpoint: {ckpt_path}")
            
            # Load checkpoint with appropriate device mapping
            if self.device == "gpu":
                ckpt = torch.load(ckpt_path, map_location="cuda")
                self.model = self.model.cuda()
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Load the model state dict directly
            self.model.load_state_dict(ckpt)
            logger.info(f"helmet checkpoint loaded successfully on {self.device}")
            
            # Create predictor - use VOC_CLASSES exactly like original demo
            self.predictor = Predictor(
                self.model, self.exp, VOC_CLASSES, None, None, self.device, False, False
            )
            
        except Exception as e:
            logger.error(f"failed to load helmet model: {e}")
            raise e
    
    def predict(self, img):
        if self.predictor is None:
            return None, -5
            
        try:
            outputs, img_info = self.predictor.inference(img)
            if len(outputs) > 0 and outputs[0] is not None:
                results = self._convert_to_api_format(outputs[0], img_info)
                return results, 0
                    
            return [], -4
            
        except Exception as e:
            logger.error(f"helmet prediction error: {e}")
            return None, -5
    
    def _convert_to_api_format(self, output, img_info, cls_conf=0.0):
        ratio = img_info["ratio"]
        if output is None:
            return []
            
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        # Use only class confidence for helmet detection
        scores = output[:, 5]
        
        results = []
        for i in range(len(cls)):
            class_idx = int(cls[i])
            score = float(scores[i])
            
            # Filter: only process helmet detections (class 0), ignore person detections (class 1)
            if class_idx != 0 or score <= cls_conf:
                continue
                
            x1, y1, x2, y2 = bboxes[i].tolist()
            width_px = x2 - x1
            height_px = y2 - y1
            cx = x1 + width_px / 2
            cy = y1 + height_px / 2
            
            # normalize coordinates
            img_width = img_info["width"]
            img_height = img_info["height"]
            cxn = cx / img_width
            cyn = cy / img_height
            width_n = width_px / img_width
            height_n = height_px / img_height
            left_n = cxn - width_n / 2
            top_n = cyn - height_n / 2
            
            class_name = self.predictor.cls_names[class_idx] if class_idx < len(self.predictor.cls_names) else f"class_{class_idx}"
            
            results.append({
                "score": score,
                "class": class_idx,
                "class_name": class_name,
                "location": {
                    "left": left_n,
                    "top": top_n,
                    "width": width_n,
                    "height": height_n
                }
            })
        
        return results

class HelmetService:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.helmet_detector = HelmetPredictor()
    
    def predict_helmet(self, img):
        try:
            # First detect persons
            persons = self.person_detector.detect_persons(img)
            logger.info(f"detected {len(persons)} persons")
            if not persons:
                return [], -4
            
            # Process each person separately
            final_results = []
            img_height, img_width = img.shape[:2]
            
            for person_idx, person in enumerate(persons):
                person_bbox = person["bbox"]
                person_conf = person["confidence"]
                logger.info(f"processing person {person_idx}: bbox={person_bbox}, conf={person_conf:.3f}")
                
                # Crop person region from image with some padding
                x1, y1, x2, y2 = person_bbox
                # Add padding (10% of bbox size)
                padding_x = int((x2 - x1) * 0.1)
                padding_y = int((y2 - y1) * 0.1)
                crop_x1 = max(0, x1 - padding_x)
                crop_y1 = max(0, y1 - padding_y)
                crop_x2 = min(img_width, x2 + padding_x)
                crop_y2 = min(img_height, y2 + padding_y)
                
                # Crop the person region
                person_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                logger.info(f"cropped person {person_idx} region: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})")
                
                if person_crop.size == 0:
                    logger.warning(f"empty crop for person {person_idx}, skipping")
                    continue
                
                # Detect helmets in the cropped person region
                helmet_results, helmet_errno = self.helmet_detector.predict(person_crop)
                logger.info(f"helmet detection in person {person_idx} crop: errno={helmet_errno}, results={len(helmet_results) if helmet_results else 0}")
                
                # Check for helmet detection in the cropped region
                has_helmet = False
                max_helmet_score = 0.0
                
                # TEMPORARY DEBUG: Log all detection details
                logger.info(f"[DEBUG] person {person_idx} helmet detection details:")
                logger.info(f"  - errno: {helmet_errno}")
                logger.info(f"  - results count: {len(helmet_results) if helmet_results else 0}")
                
                if helmet_errno == 0 and helmet_results:
                    for helmet_idx, helmet_item in enumerate(helmet_results):
                        logger.info(f"  helmet result {helmet_idx}: class={helmet_item['class']}, class_name='{helmet_item['class_name']}', score={helmet_item['score']:.3f}")
                        # Check for class 0 which is "hat" according to VOC_CLASSES
                        if helmet_item["class"] == 0:
                            has_helmet = True
                            max_helmet_score = max(max_helmet_score, helmet_item["score"])
                            logger.warning(f"  ⚠️ HELMET DETECTED for person {person_idx}, score: {helmet_item['score']:.3f} - Please verify if this is correct!")
                        else:
                            logger.info(f"  Non-helmet detection (class {helmet_item['class']}), ignoring")
                else:
                    logger.info(f"  No helmet detections in person {person_idx} crop")
                
                # Calculate helmet violation score
                if has_helmet:
                    violation_score = 0.0  # helmet detected = no violation
                    logger.warning(f"person {person_idx} has helmet detected, violation_score: {violation_score:.3f} (helmet detected = safe)")
                else:
                    violation_score = 1.0  # No helmet detected = maximum violation
                    logger.info(f"person {person_idx} no helmet detected, violation_score: 1.0 (maximum danger)")
                
                # Normalize person bbox coordinates
                person_left = person_bbox[0] / img_width
                person_top = person_bbox[1] / img_height
                person_width = (person_bbox[2] - person_bbox[0]) / img_width
                person_height = (person_bbox[3] - person_bbox[1]) / img_height
                
                final_results.append({
                    "score": violation_score,
                    "person_confidence": person_conf,
                    "has_helmet": has_helmet,
                    "helmet_score": max_helmet_score if has_helmet else 0.0,
                    "location": {
                        "left": person_left,
                        "top": person_top,
                        "width": person_width,
                        "height": person_height
                    }
                })
            
            logger.info(f"final helmet results: {len(final_results)} persons processed")
            return final_results, 0
            
        except Exception as e:
            logger.error(f"helmet safety detection error: {e}")
            return None, -5

# Initialize helmet service
helmet_service = HelmetService()

# Create Flask app
app = Flask(__name__)

def validate_img_format():
    """Validate and decode base64 image from request"""
    req = None
    try:
        req = request.json              
    except Exception:         
        return None, -1
        
    if not req or not req.get('image'):         
        return None, -2
    
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
        return None, -3

@app.route('/predict', methods=['POST'])
def predict():
    """Helmet detection endpoint"""
    img, errno = validate_img_format()
    if img is None:
        return jsonify({
            "log_id": str(uuid4()),
            "errno": errno,
            "err_msg": "validation error",
            "results": []
        })
    
    # Inference
    results, errno = helmet_service.predict_helmet(img)
    return jsonify({
        "log_id": str(uuid4()),
        "errno": errno,
        "err_msg": "success" if errno == 0 else "prediction error",
        "model_name": "helmet_detection",
        "results": results if results else []
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "helmet_detection"})

# WSGI application for gunicorn
def create_app():
    return app

if __name__ == '__main__':
    logger.info("starting helmet safety detection service on port 8902...")
    app.run(host='0.0.0.0', port=8902, debug=False)
