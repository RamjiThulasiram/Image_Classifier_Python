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

parser.add_argument('--dir', type=str, default='/flowers/',
                    help='path to folder of images')
parser.add_argument('--arch', action='store', default='vgg16',
                    help='chosen pretrained model')
parser.add_argument('--epochs', type=int, default=10,
                    help='No.of Epochs for Training')
parser.add_argument('--drop', type=float, default=0.5,
                    help='Dropout rate for Training')
parser.add_argument('--savedir', type=str, default='./checkpoint.pth',
                    help='Save directory')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate for Training')
parser.add_argument('--hidden1', type=int, default=256,
                    help='First hidden layer for Training')
parser.add_argument('--hidden2', type=int, default=128,
                    help='Second hidden layer for Training')
parser.add_argument('--device', type=str, default='cuda',
                    help='Which one you wanna work with?? - cpu or cuda')

train=parser.parse_args()

trainloader,testloader,validloader,train_data = main.DataLoader(train.dir)

model,criterion,optimizer=main.Neural_Network(train.arch,train.drop,train.hidden1,train.hidden2)

main.do_deep_learning(model,train.device,trainloader,testloader,validloader,train.epochs,20, criterion, optimizer)

main.save_checkpoint(model,train.savedir,train.arch,train.drop,train.lr,train.epochs,train.hidden1,train.hidden2,train_data,optimizer)

print("\n\nThe Model is fully trained and Saved")
