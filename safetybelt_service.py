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

# Add safetybelt model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
safetybelt_yolo_path = os.path.join(script_dir, "YOLO-main-safetybelt")
sys.path.insert(0, safetybelt_yolo_path)

# Import YOLOX modules
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

def get_device():
    """Auto-detect device: prefer CUDA, fallback to CPU"""
    if torch.cuda.is_available():
        device = "gpu"  # Use "gpu" for YOLOX compatibility
        logger.info(f"using GPU device: {torch.cuda.get_device_name()}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    return device

class PersonDetector:
    def __init__(self):
        self.model = None
        self.device = get_device()
        self._load_model()
    
    def _load_model(self):
        try:
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

class SafetyBeltPredictor:
    def __init__(self):
        self.model = None
        self.exp = None
        self.predictor = None
        self.device = get_device()
        self._load_model()
    
    def _load_model(self):
        try:
            # Get experiment configuration
            exp_file = os.path.join(safetybelt_yolo_path, "exps/example/yolox_voc/yolox_voc_s.py")
            self.exp = get_exp(exp_file, None)
            
            # Get model
            self.model = self.exp.get_model()
            logger.info(f"safetybelt model summary: {get_model_info(self.model, self.exp.test_size)}")
            
            # Set evaluation mode
            self.model.eval()
            
            # Load checkpoint
            ckpt_path = os.path.join(safetybelt_yolo_path, "weights/best_ckpt.pth")
            logger.info(f"loading safetybelt checkpoint: {ckpt_path}")
            
            # Load checkpoint with appropriate device mapping
            if self.device == "gpu":
                ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
                self.model = self.model.cuda()
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            # Handle training checkpoint format (has "model" key)
            if "model" in ckpt:
                # This is a training checkpoint, extract the model weights
                model_weights = ckpt["model"]
                logger.info("extracted model weights from training checkpoint")
            else:
                # Direct model weights
                model_weights = ckpt
                logger.info("using direct model weights")
            
            # Load the model state dict
            self.model.load_state_dict(model_weights)
            logger.info(f"safetybelt checkpoint loaded successfully on {self.device}")
            
            # Create predictor - use VOC_CLASSES exactly like original demo
            device_str = self.device
            self.predictor = Predictor(
                self.model, self.exp, VOC_CLASSES, None, None, device_str, False, False
            )
            
        except Exception as e:
            logger.error(f"failed to load safetybelt model: {e}")
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
            logger.error(f"safetybelt prediction error: {e}")
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
                class_name = self.predictor.cls_names[class_idx] if class_idx < len(self.predictor.cls_names) else f"class_{class_idx}"
                
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
            logger.info("safetybelt infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

class SafetyBeltService:
    def __init__(self):
        self.person_detector = PersonDetector()
        self.safetybelt_detector = SafetyBeltPredictor()
    
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
    
    def predict_safetybelt_compliance(self, img):
        try:
            # For safetybelt model, we don't need person detection first
            # The model directly classifies person states: illegal(0) vs compliant(1)
            
            # Detect using safetybelt model directly
            safetybelt_results, safetybelt_errno = self.safetybelt_detector.predict(img)
            logger.info(f"safetybelt detection errno: {safetybelt_errno}, results count: {len(safetybelt_results) if safetybelt_results else 0}")
            
            if safetybelt_errno != 0 or not safetybelt_results:
                logger.info("no safetybelt detection results")
                return [], -4
            
            # Log all safetybelt detection results
            for i, item in enumerate(safetybelt_results):
                logger.info(f"safetybelt result {i}: class={item['class']}, class_name='{item['class_name']}', score={item['score']:.3f}")
            
            # Process each detection result
            final_results = []
            img_height, img_width = img.shape[:2]
            
            for result_idx, safetybelt_item in enumerate(safetybelt_results):
                class_idx = safetybelt_item["class"]
                class_name = safetybelt_item["class_name"]
                detection_score = safetybelt_item["score"]
                
                logger.info(f"processing safetybelt result {result_idx}: class={class_idx}, class_name='{class_name}', score={detection_score:.3f}")
                
                # Calculate violation score based on class
                if class_idx == 0:  # "illegal" - person not wearing safetybelt
                    violation_score = detection_score  # Use detection score directly
                    has_safetybelt = False
                    safetybelt_score = 0.0
                    logger.info(f"illegal detection (no safetybelt), violation_score: {violation_score:.3f}")
                elif class_idx == 1:  # "compliant" - person wearing safetybelt
                    violation_score = 1.0 - detection_score  # Invert score
                    has_safetybelt = True
                    safetybelt_score = detection_score
                    logger.info(f"compliant detection (has safetybelt), violation_score: {violation_score:.3f}")
                else:
                    logger.warning(f"unknown class {class_idx}, skipping")
                    continue
                
                # Get normalized coordinates from detection
                location = safetybelt_item["location"]
                
                final_results.append({
                    "score": violation_score,
                    "person_confidence": 1.0,  # No separate person detection
                    "has_safetybelt": has_safetybelt,
                    "safetybelt_score": safetybelt_score,
                    "location": {
                        "left": location["left"],
                        "top": location["top"],
                        "width": location["width"],
                        "height": location["height"]
                    }
                })
            
            logger.info(f"final safetybelt results: {len(final_results)} detections processed")
            return final_results, 0
            
        except Exception as e:
            logger.error(f"safety belt detection error: {e}")
            return None, -5

# Initialize safety belt service
safetybelt_service = SafetyBeltService()

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
    """Safety belt compliance detection endpoint"""
    img, errno = validate_img_format()
    if img is None:
        return jsonify({
            "log_id": str(uuid4()),
            "errno": errno,
            "err_msg": "validation error",
            "results": []
        })
    
    # Inference
    results, errno = safetybelt_service.predict_safetybelt_compliance(img)
    return jsonify({
        "log_id": str(uuid4()),
        "errno": errno,
        "err_msg": "success" if errno == 0 else "prediction error",
        "model_name": "safetybelt_compliance",
        "results": results if results else []
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "safetybelt_compliance"})

# WSGI application for gunicorn
def create_app():
    return app

if __name__ == '__main__':
    logger.info("starting safety belt compliance detection service on port 8903...")
    app.run(host='0.0.0.0', port=8903, debug=False)
