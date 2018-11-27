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
IMAGE_PATH = 'E:/病虫害/病虫害/'
IMAGE_FORMATS = ['jpg', 'jpeg']  # 不区分大小写
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def Split_Dataset():
    sub_dirs = [x[0] for x in os.walk(IMAGE_PATH)]
    sub_dirs.pop(0)

    train_list = []
    valid_list = []
    test_list = []

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for image_format in IMAGE_FORMATS:
            file_glob = os.path.join(IMAGE_PATH, dir_name, '*.' + image_format)
            file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            chance = np.random.randint(100)
            if chance < VALIDATION_PERCENTAGE:
                valid_list.append([file_name, int(dir_name)])
            elif chance < (VALIDATION_PERCENTAGE + TEST_PERCENTAGE):
                test_list.append([file_name, int(dir_name)])
            else:
                train_list.append([file_name, int(dir_name)])
    return (train_list, valid_list, test_list)


class Classify_Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_raw = cv2.imread(self.data_list[idx][0], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        cv2.cvtColor(image_raw, )
        image = cv2.resize(image, (1366, 1024))     # resize: width * height
        image = image.transpose((2, 0, 1))  # numpy image -> torch image
        image = torch.from_numpy(image).float()

        sample = {'image': image,
                  'label': self.data_list[idx][1], 
                  'filename': self.data_list[idx][0]}
        return sample


def main():
    [train_list, valid_list, test_list] = Split_Dataset()
    train_dataset = Classify_Dataset(train_list)
    valid_dataset = Classify_Dataset(valid_list)
    test_dataset = Classify_Dataset(test_list)

    print('end')


if __name__ == "__main__":
    main()
