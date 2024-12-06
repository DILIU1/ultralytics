# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:15:24 2023

@author: 10417
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import numpy
import torch
from PIL import Image
import numpy as np
# def write_pytorch_data(output_path, data, data_name_list):
#     """
#     Save the data of Pytorch needed to align TNN model.

#     The input and output names of pytorch model and onnx model may not match,
#     you can use Netron to visualize the onnx model to determine the data_name_list.

#     The following example converts ResNet50 to onnx model and saves input and output:
#     >>> from torchvision.models.resnet import resnet50
#     >>> model = resnet50(pretrained=False).eval()
#     >>> input_data = torch.randn(1, 3, 224, 224)
#     >>> input_names, output_names = ["input"], ["output"]
#     >>> torch.onnx.export(model, input_data, "ResNet50.onnx", input_names=input_names, output_names=output_names)
#     >>> with torch.no_grad():
#     ...     output_data = model(input_data)
#     ...
#     >>> write_pytorch_data("input.txt", input_data, input_names)
#     >>> write_pytorch_data("output.txt", output_data, output_names)

#     :param output_path: Path to save data.
#     :param data: The input or output data of Pytorch model.
#     :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
#     :return:
#     """

#     if type(data) is not list and type(data) is not tuple:
#         data = [data, ]
#     assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
#     with open(output_path, "w") as f:
#         f.write("{}\n" .format(len(data)))
#         for name, data in zip(data_name_list, data):
#             # 如果 data 是 PyTorch 张量，先将其移到 CPU 上
#             data = data
#             shape = data.shape
#             description = "{} {} ".format(name, len(shape))
#             for dim in shape:
#                 description += "{} ".format(dim)
#             data_type = 0 if data.dtype == np.float32 else 3
#             fmt = "%0.6f" if data_type == 0 else "%i"
#             description += "{}".format(data_type)
#             f.write(description + "\n")
#             np.savetxt(f, data.reshape(-1), fmt=fmt)
            
# # 1. 打开图像
# image_path = r'F:\git_all\ultralytics\img_cell_20231005_103855_downsampled.png'
# image = Image.open(image_path)

# # 2. 确保图像是 RGB 模式
# image = image.convert('RGB')

# # 3. 调整图像大小
# # 将图像调整为 640x640 的大小
# image = image.resize((640, 640))

# # 4. 将图像转换为 NumPy 数组
# image_array = np.array(image)

# # 5. 确保图像是 RGB 图像并且通道顺序是 (H, W, C)
# # image_array 现在的形状是 (640, 640, 3)

# # 6. 调整通道顺序（将 HWC 转为 CHW）
# image_array = np.transpose(image_array, (2, 0, 1))

# # 7. 添加批量维度
# # 现在的形状是 (3, 640, 640)，我们需要将其转换为 (1, 3, 640, 640)
# image_array = np.expand_dims(image_array, axis=0)

# # 输出图像的形状
# print(image_array.shape)  # 应该是 (1, 3, 640, 640)


# model=r'F:\git_all\ultralytics\best.pt'



# # Load a pretrained YOLOv8n model
# model = YOLO(model)


# # # Run inference on 'bus.jpg' with arguments
# # output_data=model.predict(image_path, save=True)

# # Export the model
# model.export(task="detect",format="onnx",imgsz=640, opset=12, optimize=True,simplify=True)

# write_pytorch_data("input.txt", image_array, input_names)
# write_pytorch_data("output.txt", output_data, output_names)

from PIL import Image

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO(r"F:\git_all\ultralytics\runs\detect\train53\weights\best.pt")

# Run inference on 'bus.jpg'
results = model([r"G:\东南大学记录\刘迪写作\2024brainInfor\image\卵巢癌HE染色\1514969 05_tile_163.png"])  # results list

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk