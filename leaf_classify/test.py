import glob
import os
#import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#mport torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import cv2

DATA_PATH = './sample/'
OUTPUT_PATH = './output/'
IMAGE_FORMAT = 'jpg'

print('当前路径为：' + os.path.abspath('.'))

image_list = []
image_glob = os.path.join(DATA_PATH, '*.' + IMAGE_FORMAT)
image_list.extend(glob.glob(image_glob))

f = open(OUTPUT_PATH + 'out.txt', 'w')
for i in range(len(image_list)):    
    f.write(image_list[i] + '\n')
f.close()

input('按任意键退出')


