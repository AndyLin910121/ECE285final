import torch
from torchvision import models
import os
from glob import glob
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
#from img2vec_pytorch import Img2Vec
import pandas as pd
from ultralytics import YOLO

# Model
# Load the pretrained model
# resnet_model = models.resnet18(pretrained=True)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
#  # or yolov5m, yolov5l, yolov5x, custom

# # Images

# img_path =  './Images/'# or file, Path, PIL, OpenCV, numpy, list
# #output_path = '/media/ashish-j/B/wheat_detection/flick_data/embeddings'
# count=0
# for file in os.listdir(img_path):
# 	img = os.path.join(img_path,file)
# 	results = model(img)
# 	results.crop(save_dir='./detection_data/'+str(img)+'/')
	
	
    
print("start!!!");
resnet_model = models.resnet18(pretrained=True)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # or yolov8m, yolov8l, yolov8x, custom

# Path to the images directory
img_path = './Images/'

# Directory to save the cropped images
save_dir = './detection_data/'

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Process each image in the directory
for file in os.listdir(img_path):
    # Create the full image path
    img = os.path.join(img_path, file)
   
    # Perform detection
    results = model(img)
   
    # Save the crops
    for i, crop in enumerate(results):
        crop.save(os.path.join(save_dir, f"{os.path.splitext(file)[0]}_crop_{i}.jpg"))

    print(f'Cropped images saved for {file}')
print("end!!!");

