#!/usr/bin/env python3
import os
import sys
import base64
import binascii
import numpy as np
import cv2
import torch
import time
from flask import Flask, request, jsonify
from loguru import logger
from uuid import uuid4
from ultralytics import YOLO

# Add helmet model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
helmet_yolo_path = os.path.join(script_dir, "YOLO-main-helmet")
sys.path.insert(0, helmet_yolo_path)

# Import YOLOX modules
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

class PersonDetector:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            model_path = os.path.join(script_dir, "models", "yolo11s.pt")
            logger.info(f"loading person detection model: {model_path}")
            self.model = YOLO(model_path)
            logger.info("person detection model loaded successfully")
        except Exception as e:
            logger.error(f"failed to load person detection model: {e}")
            raise e
    
    def detect_persons(self, img, conf_threshold=0.5):
        try:
            results = self.model.predict(img, conf=conf_threshold, classes=[0], verbose=False)
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
        self._load_model()
    
    def _load_model(self):
        try:
            # Get experiment configuration
            exp_file = os.path.join(helmet_yolo_path, "exps/example/yolox_voc/yolox_voc_s.py")
            self.exp = get_exp(exp_file, None)
            
            # Get model
            self.model = self.exp.get_model()
            logger.info(f"helmet model summary: {get_model_info(self.model, self.exp.test_size)}")
            
            # Use CPU for compatibility
            self.model.eval()
            
            # Load checkpoint
            ckpt_path = os.path.join(helmet_yolo_path, "weights/best.pth")
            logger.info(f"loading helmet checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Load model state dict
            self.model.load_state_dict(ckpt)
            logger.info("helmet checkpoint loaded successfully")
            
            # Create predictor
            self.predictor = Predictor(
                self.model, self.exp, ["person", "helmet"], None, None, "cpu", False, False
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
    
    def _convert_to_api_format(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        if output is None:
            return []
            
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        
        results = []
        for i in range(len(cls)):
            if float(scores[i]) > cls_conf:
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
                
                class_idx = int(cls[i])
                class_name = "person" if class_idx == 0 else "helmet"
                
                results.append({
                    "score": float(scores[i]),
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

class Predictor:
    def __init__(self, model, exp, cls_names=VOC_CLASSES, trt_file=None, decoder=None, device="cpu", fp16=False, legacy=False):
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

class HelmetSafetyService:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.helmet_detector = HelmetPredictor()
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def predict_helmet_safety(self, img):
        try:
            # First detect persons
            persons = self.person_detector.detect_persons(img)
            if not persons:
                return [], -4
            
            # Then detect helmets
            helmet_results, helmet_errno = self.helmet_detector.predict(img)
            
            # Process results for each person
            final_results = []
            img_height, img_width = img.shape[:2]
            
            for person in persons:
                person_bbox = person["bbox"]
                person_conf = person["confidence"]
                
                # Check if any helmet overlaps with this person
                has_helmet = False
                max_helmet_score = 0.0
                
                if helmet_errno == 0 and helmet_results:
                    for helmet_item in helmet_results:
                        if helmet_item["class_name"] == "helmet":
                            # Convert normalized coordinates back to pixel coordinates
                            loc = helmet_item["location"]
                            helmet_x1 = int((loc["left"]) * img_width)
                            helmet_y1 = int((loc["top"]) * img_height)
                            helmet_x2 = int((loc["left"] + loc["width"]) * img_width)
                            helmet_y2 = int((loc["top"] + loc["height"]) * img_height)
                            helmet_bbox = [helmet_x1, helmet_y1, helmet_x2, helmet_y2]
                            
                            # Check IoU overlap
                            iou = self.calculate_iou(person_bbox, helmet_bbox)
                            if iou > 0.1:  # If there's some overlap
                                has_helmet = True
                                max_helmet_score = max(max_helmet_score, helmet_item["score"])
                
                # Calculate helmet violation score (1 - helmet_score)
                if has_helmet:
                    violation_score = 1.0 - max_helmet_score
                else:
                    violation_score = 1.0  # No helmet detected = maximum violation
                
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
            
            return final_results, 0
            
        except Exception as e:
            logger.error(f"helmet safety detection error: {e}")
            return None, -5

# Initialize helmet safety service
helmet_safety_service = HelmetSafetyService()

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
    """Helmet safety detection endpoint"""
    img, errno = validate_img_format()
    if img is None:
        return jsonify({
            "log_id": str(uuid4()),
            "errno": errno,
            "err_msg": "validation error",
            "results": []
        })
    
    # Inference
    results, errno = helmet_safety_service.predict_helmet_safety(img)
    return jsonify({
        "log_id": str(uuid4()),
        "errno": errno,
        "err_msg": "success" if errno == 0 else "prediction error",
        "model_name": "helmet_safety",
        "results": results if results else []
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "helmet_safety"})

if __name__ == '__main__':
    logger.info("starting helmet safety detection service on port 8902...")
    app.run(host='0.0.0.0', port=8902, debug=False)
