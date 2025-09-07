#!/usr/bin/env python3

import sys
import os
import torch
from loguru import logger

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_torch_load():
    """Test torch.load with weights_only=False"""
    
    # Test different checkpoint files
    test_files = [
        "YOLO-main-fire/weights/best.pth",
        "YOLO-main-helmet/weights/best.pth", 
        "YOLO-main-safetybelt/weights/best_ckpt.pth"
    ]
    
    for ckpt_file in test_files:
        if os.path.exists(ckpt_file):
            logger.info(f"Testing {ckpt_file}")
            try:
                # Test with weights_only=True (should fail)
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
                logger.success(f"{ckpt_file} loaded with weights_only=True")
            except Exception as e:
                logger.warning(f"{ckpt_file} failed with weights_only=True: {e}")
                
            try:
                # Test with weights_only=False (should work)
                ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                logger.success(f"{ckpt_file} loaded with weights_only=False")
            except Exception as e:
                logger.error(f"{ckpt_file} failed with weights_only=False: {e}")
        else:
            logger.warning(f"{ckpt_file} not found")

if __name__ == "__main__":
    test_torch_load()
