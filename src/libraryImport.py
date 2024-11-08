#import Libraraies and HyperParamters

#Import Libraries
import time
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from os import EX_PROTOCOL




#Define Hyperparameters -

batchSize = 128
#batchSizeTest = 1000
maxEpoch = 10
learningRate = 0.05
criterion = nn.CrossEntropyLoss()



