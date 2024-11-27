# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:15:24 2023

@author: 10417
"""

from ultralytics import YOLO
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load an official model


# # Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image