#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# <a align="left" href="https://ultralytics.com/yolov3" target="_blank">
# <img width="1024", src="https://user-images.githubusercontent.com/26833433/99805971-90f66b80-2b3d-11eb-80eb-8b45a15cb68e.jpg"></a>
# 
# This is the **official YOLOv3 🚀 notebook** by **Ultralytics**, and is freely available for redistribution under the [GPL-3.0 license](https://choosealicense.com/licenses/gpl-3.0/). 
# For more information please visit https://github.com/ultralytics/yolov3 and https://ultralytics.com. Thank you!

# # Setup
# 
# Clone repo, install dependencies and check PyTorch and GPU.

# In[24]:


get_ipython().system('git clone https://github.com/ultralytics/yolov3  # clone')
get_ipython().run_line_magic('cd', 'yolov3')
get_ipython().run_line_magic('pip', 'install -qr requirements.txt  # install')

import torch
from yolov3 import utils
display = utils.notebook_init()  # checks


# # 1. Inference
# 
# `detect.py` runs YOLOv3 inference on a variety of sources, downloading models automatically from the [latest YOLOv3 release](https://github.com/ultralytics/yolov3/releases), and saving results to `runs/detect`. Example inference sources are:
# 
# ```shell
# python detect.py --source 0  # webcam
#                           img.jpg  # image 
#                           vid.mp4  # video
#                           path/  # directory
#                           path/*.jpg  # glob
#                           'https://youtu.be/Zgi9g1ksQHc'  # YouTube
#                           'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
# ```

# In[22]:


get_ipython().system('python detect.py --weights yolov3.pt --img 640 --conf 0.25 --source data/images')
display.Image(filename='runs/detect/exp/zidane.jpg', width=600)


# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
# <img align="left" src="https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg" width="600">

# # 2. Validate
# Validate a model's accuracy on [COCO](https://cocodataset.org/#home) val or test-dev datasets. Models are downloaded automatically from the [latest YOLOv3 release](https://github.com/ultralytics/yolov3/releases). To show results by class use the `--verbose` flag. Note that `pycocotools` metrics may be ~1% better than the equivalent repo metrics, as is visible below, due to slight differences in mAP computation.

# ## COCO val
# Download [COCO val 2017](https://github.com/ultralytics/yolov3/blob/master/data/coco.yaml) dataset (1GB - 5000 images), and test model accuracy.

# In[4]:


# Download COCO val
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')
get_ipython().system('unzip -q tmp.zip -d ../datasets && rm tmp.zip')


# In[23]:


# Run YOLOv3 on COCO val
get_ipython().system('python val.py --weights yolov3.pt --data coco.yaml --img 640 --iou 0.65 --half')


# ## COCO test
# Download [COCO test2017](https://github.com/ultralytics/yolov3/blob/master/data/coco.yaml) dataset (7GB - 40,000 images), to test model accuracy on test-dev set (**20,000 images, no labels**). Results are saved to a `*.json` file which should be **zipped** and submitted to the evaluation server at https://competitions.codalab.org/competitions/20794.

# In[ ]:


# Download COCO test-dev2017
torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017labels.zip', 'tmp.zip')
get_ipython().system('unzip -q tmp.zip -d ../datasets && rm tmp.zip')
get_ipython().system('f="test2017.zip" && curl http://images.cocodataset.org/zips/$f -o $f && unzip -q $f -d ../datasets/coco/images')


# In[ ]:


# Run YOLOv3 on COCO test
get_ipython().system('python val.py --weights yolov3.pt --data coco.yaml --img 640 --iou 0.65 --half --task test')


# # 3. Train
# 
# <p align=""><a href="https://roboflow.com/?ref=ultralytics"><img width="1000" src="https://uploads-ssl.webflow.com/5f6bc60e665f54545a1e52a5/615627e5824c9c6195abfda9_computer-vision-cycle.png"/></a></p>
# Close the active learning loop by sampling images from your inference conditions with the `roboflow` pip package
# <br><br>
# 
# Train a YOLOv3 model on the [COCO128](https://www.kaggle.com/ultralytics/coco128) dataset with `--data coco128.yaml`, starting from pretrained `--weights yolov3.pt`, or from randomly initialized `--weights '' --cfg yolov3yaml`.
# 
# - **Pretrained [Models](https://github.com/ultralytics/yolov3/tree/master/models)** are downloaded
# automatically from the [latest YOLOv3 release](https://github.com/ultralytics/yolov3/releases)
# - **[Datasets](https://github.com/ultralytics/yolov3/tree/master/data)** available for autodownload include: [COCO](https://github.com/ultralytics/yolov3/blob/master/data/coco.yaml), [COCO128](https://github.com/ultralytics/yolov3/blob/master/data/coco128.yaml), [VOC](https://github.com/ultralytics/yolov3/blob/master/data/VOC.yaml), [Argoverse](https://github.com/ultralytics/yolov3/blob/master/data/Argoverse.yaml), [VisDrone](https://github.com/ultralytics/yolov3/blob/master/data/VisDrone.yaml), [GlobalWheat](https://github.com/ultralytics/yolov3/blob/master/data/GlobalWheat2020.yaml), [xView](https://github.com/ultralytics/yolov3/blob/master/data/xView.yaml), [Objects365](https://github.com/ultralytics/yolov3/blob/master/data/Objects365.yaml), [SKU-110K](https://github.com/ultralytics/yolov3/blob/master/data/SKU-110K.yaml).
# - **Training Results** are saved to `runs/train/` with incrementing run directories, i.e. `runs/train/exp2`, `runs/train/exp3` etc.
# <br><br>
# 

# In[ ]:


# Tensorboard  (optional)
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs/train')


# In[ ]:


# Weights & Biases  (optional)
get_ipython().run_line_magic('pip', 'install -q wandb')
import wandb
wandb.login()


# In[21]:


# Train YOLOv3 on COCO128 for 3 epochs
get_ipython().system('python train.py --img 640 --batch 16 --epochs 3 --data coco128.yaml --weights yolov3.pt --cache')


# # 4. Visualize

# ## Weights & Biases Logging 🌟 NEW
# 
# [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_notebook) (W&B) is now integrated with YOLOv3 for real-time visualization and cloud logging of training runs. This allows for better run comparison and introspection, as well improved visibility and collaboration for teams. To enable W&B `pip install wandb`, and then train normally (you will be guided through setup on first use). 
# 
# During training you will see live updates at [https://wandb.ai/home](https://wandb.ai/home?utm_campaign=repo_yolo_notebook), and you can create and share detailed [Reports](https://wandb.ai/glenn-jocher/yolov5_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY) of your results. For more information see the [YOLOv5 Weights & Biases Tutorial](https://github.com/ultralytics/yolov5/issues/1289). 
# 
# <p align="left"><img width="900" alt="Weights & Biases dashboard" src="https://user-images.githubusercontent.com/26833433/135390767-c28b050f-8455-4004-adb0-3b730386e2b2.png"></p>

# ## Local Logging
# 
# All results are logged by default to `runs/train`, with a new experiment directory created for each new training as `runs/train/exp2`, `runs/train/exp3`, etc. View train and val jpgs to see mosaics, labels, predictions and augmentation effects. Note an Ultralytics **Mosaic Dataloader** is used for training (shown below), which combines 4 images into 1 mosaic during training.
# 
# > <img src="https://user-images.githubusercontent.com/26833433/131255960-b536647f-7c61-4f60-bbc5-cb2544d71b2a.jpg" width="700">  
# `train_batch0.jpg` shows train batch 0 mosaics and labels
# 
# > <img src="https://user-images.githubusercontent.com/26833433/131256748-603cafc7-55d1-4e58-ab26-83657761aed9.jpg" width="700">  
# `test_batch0_labels.jpg` shows val batch 0 labels
# 
# > <img src="https://user-images.githubusercontent.com/26833433/131256752-3f25d7a5-7b0f-4bb3-ab78-46343c3800fe.jpg" width="700">  
# `test_batch0_pred.jpg` shows val batch 0 _predictions_
# 
# Training results are automatically logged to [Tensorboard](https://www.tensorflow.org/tensorboard) and [CSV](https://github.com/ultralytics/yolov5/pull/4148) as `results.csv`, which is plotted as `results.png` (below) after training completes. You can also plot any `results.csv` file manually:
# 
# ```python
# from utils.plots import plot_results 
# plot_results('path/to/results.csv')  # plot 'results.csv' as 'results.png'
# ```
# 
# <img align="left" width="800" alt="COCO128 Training Results" src="https://user-images.githubusercontent.com/26833433/126906780-8c5e2990-6116-4de6-b78a-367244a33ccf.png">

# # Environments
# 
# YOLOv3 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):
# 
# - **Google Colab and Kaggle** notebooks with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov3/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov3"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
# - **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
# - **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/AWS-Quickstart)
# - **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart) <a href="https://hub.docker.com/r/ultralytics/yolov3"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov3?logo=docker" alt="Docker Pulls"></a>
# 

# # Status
# 
# ![CI CPU testing](https://github.com/ultralytics/yolov3/workflows/CI%20CPU%20testing/badge.svg)
# 
# If this badge is green, all [YOLOv3 GitHub Actions](https://github.com/ultralytics/yolov3/actions) Continuous Integration (CI) tests are currently passing. CI tests verify correct operation of YOLOv3 training ([train.py](https://github.com/ultralytics/yolov3/blob/master/train.py)), testing ([val.py](https://github.com/ultralytics/yolov3/blob/master/val.py)), inference ([detect.py](https://github.com/ultralytics/yolov3/blob/master/detect.py)) and export ([export.py](https://github.com/ultralytics/yolov3/blob/master/export.py)) on MacOS, Windows, and Ubuntu every 24 hours and on every commit.
# 

# # Appendix
# 
# Optional extras below. Unit tests validate repo functionality and should be run on any PRs submitted.
# 

# In[ ]:


# Reproduce
for x in 'yolov3', 'yolov3-spp', 'yolov3-tiny':
  get_ipython().system('python val.py --weights {x}.pt --data coco.yaml --img 640 --task speed  # speed')
  get_ipython().system('python val.py --weights {x}.pt --data coco.yaml --img 640 --conf 0.001 --iou 0.65  # mAP')


# In[ ]:


# PyTorch Hub
import torch

# Model
model = torch.hub.load('ultralytics/yolov3', 'yolov3')

# Images
dir = 'https://ultralytics.com/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batch of images

# Inference
results = model(imgs)
results.print()  # or .show(), .save()


# In[ ]:


# CI Checks
%%shell
export PYTHONPATH="$PWD"  # to run *.py. files in subdirectories
rm -rf runs  # remove runs/
for m in yolov3-tiny; do  # models
  python train.py --img 64 --batch 32 --weights $m.pt --epochs 1 --device 0  # train pretrained
  python train.py --img 64 --batch 32 --weights '' --cfg $m.yaml --epochs 1 --device 0  # train scratch
  for d in 0 cpu; do  # devices
    python val.py --weights $m.pt --device $d # val official
    python val.py --weights runs/train/exp/weights/best.pt --device $d # val custom
    python detect.py --weights $m.pt --device $d  # detect official
    python detect.py --weights runs/train/exp/weights/best.pt --device $d  # detect custom
  done
  python hubconf.py  # hub
  python models/yolo.py --cfg $m.yaml  # build PyTorch model
  python models/tf.py --weights $m.pt  # build TensorFlow model
  python export.py --img 64 --batch 1 --weights $m.pt --include torchscript onnx  # export
done


# In[ ]:


# Profile
from utils.torch_utils import profile

m1 = lambda x: x * torch.sigmoid(x)
m2 = torch.nn.SiLU()
results = profile(input=torch.randn(16, 3, 640, 640), ops=[m1, m2], n=100)


# In[ ]:


# Evolve
get_ipython().system('python train.py --img 640 --batch 64 --epochs 100 --data coco128.yaml --weights yolov3.pt --cache --noautoanchor --evolve')
get_ipython().system('d=runs/train/evolve && cp evolve.* $d && zip -r evolve.zip $d && gsutil mv evolve.zip gs://bucket  # upload results (optional)')


# In[ ]:


# VOC
for b, m in zip([24, 24, 64], ['yolov3', 'yolov3-spp', 'yolov3-tiny']):  # zip(batch_size, model)
  get_ipython().system('python train.py --batch {b} --weights {m}.pt --data VOC.yaml --epochs 50 --cache --img 512 --nosave --hyp hyp.finetune.yaml --project VOC --name {m}')

