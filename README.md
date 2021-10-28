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

# Train & Validation
First, you need to download weights of ResNet50 pretrained on ImageNet database.
- Downlod the weights from this website (https://download.pytorch.org/models/resnet50-0676ba61.pth)
- rename the .pth file as "resnet50.pth" and put it in the "model" folder

Second, you need to download the DF2K dataset.
- DF2K is a merged training dataset consisting of 800 DIV2K training images and 2560 Flickr2K training images
- DIV2K download link: https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
- Flickr2K download link: https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar 
- After merging the datasets (DIV2K & Flickr2K), the DF2K dataset should be composed as below

DF2K
  L DF2K_HR
  L DF2K_LR_bicubic
  L DF2K_LR_unknown

- set the database path in "train.py" (It is represented as "db_path" in "train.py")


After those settings, you can run the train & validation code by running "train.py"
- python3 train.py (execution code)
- This code works on single GPU. If you want to train this code in muti-gpu, you need to change this code
- Options are all included in "train.py". So you should change the variable "config" in "train.py"
![image](https://user-images.githubusercontent.com/77471764/138195607-cf7165a1-dd64-4031-b1ab-872012f7046a.png)

Belows are the validation performance on KonIQ-10k database (I'm still training the code, so the results will be updated later)
- SRCC: 0.9023 / PLCC: 0.9232 (after training 105 epochs)
- If the codes are implemented exactly the same as the paper, the performance can be further improved

# Inference
First, you need to specify variables in "inference.py"
- dirname: root folder of test images
- checkpoint: checkpoint file (trained on KonIQ-10k dataset)
- result_score_txt: inference score will be saved on this txt file
![image](https://user-images.githubusercontent.com/77471764/138195041-3176224f-6ab6-42b1-aa61-f9ec8a1ffa96.png)

After those settings, you can run the inference code by running "inference.py"
- python3 inference.py (execution code)

# Acknolwdgements
We refer to the following website to implement the transformer (https://paul-hyun.github.io/transformer-01/)
