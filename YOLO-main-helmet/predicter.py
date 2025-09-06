import cv2
import time
import torch
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

video = cv2.VideoCapture()

while True:
    rc, image = video.read()



