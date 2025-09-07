import os
import sys
from enum import Enum
from uuid import uuid4
import torch
import time
import numpy as np
from loguru import logger
from ultralytics import YOLO
import cv2

VERSION_API = '0.0.1'

class ServiceStatus(Enum):
    SUCCESS = 0
    INVALID_CONTENT_TYPE = -1
    MISSING_IMAGE_DATA = -2
    INVALID_IMAGE_FORMAT = -3
    NO_OBJECT_DETECTED = -4
    MODEL_LOAD_ERROR = -5
    
    @staticmethod
    def stringify(errno):
        status_map = {
            ServiceStatus.SUCCESS.value: 'SUCCESS',
            ServiceStatus.INVALID_CONTENT_TYPE.value: 'INVALID_CONTENT_TYPE',
            ServiceStatus.MISSING_IMAGE_DATA.value: 'MISSING_IMAGE_DATA',
            ServiceStatus.INVALID_IMAGE_FORMAT.value: 'INVALID_IMAGE_FORMAT',
            ServiceStatus.NO_OBJECT_DETECTED.value: 'NO_OBJECT_DETECTED',
            ServiceStatus.MODEL_LOAD_ERROR.value: 'MODEL_LOAD_ERROR'
        }
        return status_map.get(errno, 'UNKNOWN_ERROR')

# Simple direct imports - no complex module isolation needed
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess

def get_predictor_class():
    """Return the simple Predictor class"""
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
                logger.info("infer time: {:.4f}s".format(time.time() - t0))
            return outputs, img_info

    return Predictor

class YOLOXDetectionService:
    def __init__(self, model_name, model_dir, exp_file, ckpt_file, device=None):
        self.model_name = model_name
        self.version = "0.0.1"
        self.model_dir = model_dir
        self.exp_file = exp_file  
        self.ckpt_file = ckpt_file
        # Auto-detect device if not specified
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        
        logger.info(f"initializing {model_name} model on device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        try:
            # Get the simple predictor class
            Predictor = get_predictor_class()
            
            # Get experiment configuration
            exp = get_exp(os.path.join(self.model_dir, self.exp_file), None)
            
            # Override num_classes based on model type
            if self.model_name == "fire":
                exp.num_classes = 1  # only fire class
            elif "helmet" in self.model_name:
                exp.num_classes = 2  # person + helmet
            elif "safetybelt" in self.model_name:
                exp.num_classes = 2  # person + safetybelt
            
            # Get model
            model = exp.get_model()
            logger.info(f"model summary for {self.model_name}: {get_model_info(model, exp.test_size)}")
            
            if self.device == "cuda":
                model.cuda()
            model.eval()
            
            # Load checkpoint
            ckpt_path = os.path.join(self.model_dir, self.ckpt_file)
            logger.info(f"loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # Load model state dict - exactly like original demo
            model.load_state_dict(ckpt)
            logger.info("loaded checkpoint done.")
            
            # Create predictor with custom class names
            cls_names = None
            if self.model_name == "fire":
                cls_names = ["fire"]
            elif "helmet" in self.model_name:
                cls_names = ["person", "helmet"] 
            elif "safetybelt" in self.model_name:
                cls_names = ["person", "safetybelt"]
            
            # Use "gpu" for device like the original demo
            device_str = "gpu" if self.device == "cuda" else "cpu"
            self.predictor = Predictor(model, exp, cls_names, None, None, device_str, False, False)
            
        except Exception as e:
            logger.error(f"failed to load {self.model_name} model: {e}")
            raise e
    
    def predict(self, img):
        if self.predictor is None:
            return None, ServiceStatus.MODEL_LOAD_ERROR.value
            
        try:
            outputs, img_info = self.predictor.inference(img)
            if len(outputs) > 0 and outputs[0] is not None:
                results = self._convert_to_api_format(outputs[0], img_info)
                if len(results) > 0:
                    return results, ServiceStatus.SUCCESS.value
                    
            return [], ServiceStatus.NO_OBJECT_DETECTED.value
            
        except Exception as e:
            logger.error(f"prediction error for {self.model_name}: {e}")
            return None, ServiceStatus.MODEL_LOAD_ERROR.value
    
    def _convert_to_api_format(self, output, img_info, cls_conf=0.35):
        """Convert YOLOX output to API format"""
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
                class_name = self.predictor.cls_names[class_idx] if class_idx < len(self.predictor.cls_names) else "unknown"
                
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
    
    def response(self, errno, results=None):
        return {
            "log_id": str(uuid4()),
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
            "api_version": VERSION_API,
            "model_version": self.version,
            "model_name": self.model_name,
            "results": results if results else []
        }

class PersonDetectionService:
    """Person detection using YOLO11"""
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            logger.info(f"loading person detection model: {self.model_path}")
            self.model = YOLO(self.model_path)
            if self.device == "cuda":
                self.model.to("cuda")
            logger.info(f"person detection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"failed to load person detection model: {e}")
            raise e
    
    def detect_persons(self, img, conf_threshold=0.5):
        """Detect persons in image and return bounding boxes"""
        try:
            results = self.model.predict(img, conf=conf_threshold, classes=[0], verbose=False)  # class 0 is person
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

class SafetyDetectionService:
    """Safety equipment detection (helmet/safetybelt) with person detection"""
    def __init__(self, model_name, model_dir, exp_file, ckpt_file, person_detector, device=None):
        self.safety_detector = YOLOXDetectionService(model_name, model_dir, exp_file, ckpt_file, device)
        self.person_detector = person_detector
        self.model_name = model_name
    
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
    
    def predict_safety(self, img):
        """Detect safety equipment on persons"""
        try:
            # First detect persons
            persons = self.person_detector.detect_persons(img)
            if not persons:
                return [], ServiceStatus.NO_OBJECT_DETECTED.value
            
            # Then detect safety equipment
            safety_results, safety_errno = self.safety_detector.predict(img)
            
            # Process results for each person
            final_results = []
            img_height, img_width = img.shape[:2]
            
            for person in persons:
                person_bbox = person["bbox"]
                person_conf = person["confidence"]
                
                # Check if any safety equipment overlaps with this person
                has_safety_equipment = False
                max_safety_score = 0.0
                
                if safety_errno == ServiceStatus.SUCCESS.value and safety_results:
                    for safety_item in safety_results:
                        # Convert normalized coordinates back to pixel coordinates
                        loc = safety_item["location"]
                        safety_x1 = int((loc["left"]) * img_width)
                        safety_y1 = int((loc["top"]) * img_height)
                        safety_x2 = int((loc["left"] + loc["width"]) * img_width)
                        safety_y2 = int((loc["top"] + loc["height"]) * img_height)
                        safety_bbox = [safety_x1, safety_y1, safety_x2, safety_y2]
                        
                        # Check IoU overlap
                        iou = self.calculate_iou(person_bbox, safety_bbox)
                        if iou > 0.1:  # If there's some overlap
                            has_safety_equipment = True
                            max_safety_score = max(max_safety_score, safety_item["score"])
                
                # Calculate safety violation score (1 - safety_score)
                if has_safety_equipment:
                    violation_score = 1.0 - max_safety_score
                else:
                    violation_score = 1.0  # No safety equipment detected = maximum violation
                
                # Normalize person bbox coordinates
                person_left = person_bbox[0] / img_width
                person_top = person_bbox[1] / img_height
                person_width = (person_bbox[2] - person_bbox[0]) / img_width
                person_height = (person_bbox[3] - person_bbox[1]) / img_height
                
                final_results.append({
                    "score": violation_score,
                    "person_confidence": person_conf,
                    "has_safety_equipment": has_safety_equipment,
                    "safety_equipment_score": max_safety_score if has_safety_equipment else 0.0,
                    "location": {
                        "left": person_left,
                        "top": person_top,
                        "width": person_width,
                        "height": person_height
                    }
                })
            
            return final_results, ServiceStatus.SUCCESS.value
            
        except Exception as e:
            logger.error(f"safety detection error for {self.model_name}: {e}")
            return None, ServiceStatus.MODEL_LOAD_ERROR.value
    
    def response(self, errno, results=None):
        return {
            "log_id": str(uuid4()),
            "errno": errno,
            "err_msg": ServiceStatus.stringify(errno),
            "api_version": VERSION_API,
            "model_version": "0.0.1",
            "model_name": self.model_name,
            "results": results if results else []
        }
