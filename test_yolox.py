#!/usr/bin/env python3
"""
Simple YOLOX installation test
"""
import sys
import os

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import YOLOX components:")

try:
    import yolox
    print(f"✓ yolox: {yolox.__file__}")
    print(f"  version: {yolox.__version__}")
except ImportError as e:
    print(f"✗ yolox: {e}")

try:
    import yolox.data
    print(f"✓ yolox.data: {yolox.data.__file__}")
except ImportError as e:
    print(f"✗ yolox.data: {e}")

try:
    from yolox.data.data_augment import ValTransform
    print("✓ yolox.data.data_augment.ValTransform imported successfully")
except ImportError as e:
    print(f"✗ yolox.data.data_augment.ValTransform: {e}")

try:
    from yolox.exp import get_exp
    print("✓ yolox.exp.get_exp imported successfully")
except ImportError as e:
    print(f"✗ yolox.exp.get_exp: {e}")

print("\nManually checking YOLO directories:")
yolo_dirs = ['/root/YOLO-main-fire', '/root/YOLO-main-helmet', '/root/YOLO-main-safetybelt']
for yolo_dir in yolo_dirs:
    if os.path.exists(yolo_dir):
        print(f"\n{yolo_dir}:")
        yolox_dir = os.path.join(yolo_dir, 'yolox')
        if os.path.exists(yolox_dir):
            print(f"  yolox dir exists: {os.listdir(yolox_dir)}")
            data_dir = os.path.join(yolox_dir, 'data')
            if os.path.exists(data_dir):
                print(f"  data dir exists: {os.listdir(data_dir)}")
            else:
                print("  data dir MISSING")
        else:
            print("  yolox dir MISSING")
    else:
        print(f"{yolo_dir} does not exist")

print("\nTrying with path manipulation:")
sys.path.insert(0, '/root/YOLO-main-fire')
try:
    from yolox.data.data_augment import ValTransform
    print("✓ ValTransform imported with path manipulation")
except ImportError as e:
    print(f"✗ ValTransform with path manipulation: {e}")
