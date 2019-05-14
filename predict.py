import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#The Entire Programming Section is stored as main.py for easier presentation
import main

import argparse

parser=argparse.ArgumentParser()

parser.add_argument('--savedir', type=str, default='/home/workspace/aipnd-project/checkpoint.pth',
                    help='checkpoint save directory')
parser.add_argument('--path', type=str, default='/flowers/',
                    help='path to folder of images')
parser.add_argument('--img_path', type=str, default='/home/workspace/aipnd-project/flowers/test/10/image_07090.jpg',
                    help='path of the testing images')
parser.add_argument('--device', type=str, default='cuda',
                    help='Which device you want to work with cpu? or cuda?')
parser.add_argument('--topk', type=int, default=5,
                    help='Top probablities that you want to display')

predict=parser.parse_args()

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

trainloader,testloader,validloader,train_data = main.DataLoader(predict.path)

model,class_to_idx = main.load_checkpoint(predict.savedir)

probs,classes = main.predict(predict.img_path, model, predict.topk,predict.device,class_to_idx,train_data)

x = probs
y = [cat_to_name[str(i)] for i in classes]

print("-- Probablities :")
a=0    
for a in range(0,predict.topk,1):
    print("Image Class : {}  \nProbablity : {}%\n".format(y[a],x[a]*100))
    a+=1

print("--The Prediction is Finished")