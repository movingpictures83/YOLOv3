import torch

# Model
model = torch.hub.load('ultralytics/yolov3', 'yolov3-spp')  # or yolov3-spp, yolov3-tiny, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

