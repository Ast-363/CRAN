# CRAN
Unofficial pytorch implementation of the paper "Context Reasoning Attention Network for Image Super-Resolution (ICCV 2021)"

This code doesn't exactly match what the paper describes.
- I delete the module "CDRR" (Section 3.2) in the code because of low training speed and performance drop (maybe I wrongly make the code for "CDRR")
- If you want to use "CDRR", just uncomment the remarks in "common.py"
![capture](https://user-images.githubusercontent.com/77471764/139164973-89448c36-70cd-48ed-8126-0fa253320d9f.PNG)
![capture1](https://user-images.githubusercontent.com/77471764/139164983-9b7fb654-6092-4800-9e43-7ce4e0556eae.PNG)

The environmental settings are described below. (I cannot gaurantee if it works on other environments)
- Pytorch=1.7.1+cu110 
- numpy=1.18.3
- cv2=4.2.0
- tqdm=4.45.0

# Train
First, you need to download weights of ResNet50 pretrained on ImageNet database.
- Downlod the weights from this website (https://download.pytorch.org/models/resnet50-0676ba61.pth)
- rename the .pth file as "resnet50.pth" and put it in the "model" folder

Second, you need to download the DF2K dataset.
- DF2K is a merged training dataset consisting of 800 DIV2K training images and 2560 Flickr2K training images
- DIV2K download link: https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
- Flickr2K download link: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar 
- After merging the datasets (DIV2K & Flickr2K), the DF2K dataset should be composed as below

![캡처](https://user-images.githubusercontent.com/77471764/139165671-b12c5b6d-3f12-4564-bdf1-83c86f688a29.PNG)

- Set the database path in "./opt/option.py" (It is represented as "dir_data")

After those settings, you can run the train code by running "train.py"
- python3 train.py --gpu_id 0 (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "./opt/option.py". So you should change the variable in "./opt/option.py"

# Inference
First, you need to specify variables in "./opt/option.py"
- dir_test: root folder of test images
- weights: checkpoint file (trained on DF2K dataset)
- results: inference results will be saved on this folder

After those settings, you can run the inference code by running "inference.py"
- python3 inference.py (execution code)

# Acknolwdgements
We refer to repos below to implement this code.
- official RCAN github (https://github.com/yulunzhang/RCAN)
- official Context-Gated Convolution github (https://github.com/XudongLinthu/context-gated-convolution)
