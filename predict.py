# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:15:24 2023

@author: 10417
"""

from ultralytics import YOLO
from ultralytics import YOLO

# 加载模型
model = YOLO(r"F:\git_all\ultralytics\runs\detect\train69\weights\best.pt")

# 使用模型进行预测
results = model(r"G:/东南大学记录/刘迪写作/2024brainInfor/image/小鼠脑DIC/dataset_20241205_164213_yolo/dataset/images/train/image_001_1_1.png")  # 预测单张图片
