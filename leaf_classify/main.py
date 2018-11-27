import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

MODEL_PATH = './model.pth'
IMAGE_PATH = 'D:/code/dataset/病虫害/'
IMAGE_PATH = 'JPG'


class Classify_Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return 0

