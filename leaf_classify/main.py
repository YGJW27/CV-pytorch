import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2

MODEL_PATH = 'model.pth'
DATA_PATH = './sample/'
OUTPUT_PATH = './output/'
IMAGE_FORMAT = 'jpg'


class Classify_Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_raw = cv2.imread(self.data_list[idx], cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1366, 1024))     # resize: width * height
        image = image.transpose((2, 0, 1))  # numpy image -> torch image
        image = torch.from_numpy(image).float()
        image = image / 255.

        sample = {'image': image,
                  'filename': self.data_list[idx]}
        return sample


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 25, 5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(25, 64, 5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(3, stride=3)
        self.conv5 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.pool5 = nn.MaxPool2d(3, stride=3)
        self.fc1 = nn.Linear(18 * 14 * 64, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test(test_dataloader, model, device):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        f = open(OUTPUT_PATH + 'out.txt', 'w')
        for batch_idx, batched_sample in enumerate(test_dataloader):
            inputs = batched_sample['image']
            inputs = inputs.to(device)

            output = model(inputs)
            prob = F.softmax(output.data, dim=1)

            _, predicted = torch.max(output.data, 1)

            filename = batched_sample['filename']

            f.write(filename[0] + '  ')
            f.write(str(predicted[0].item()) + '  ')
            f.write('{:.2f}'.format(
                (prob[0][predicted[0].item()].item())) + '\n')

            print('已完成识别: {:.0f}%'.format(
                batch_idx / len(test_dataloader.dataset) * 100))

        f.close()


def main():
    test_list = []
    test_glob = os.path.join(DATA_PATH, '*.' + IMAGE_FORMAT)
    test_list.extend(glob.glob(test_glob))

    test_dataset = Classify_Dataset(test_list)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()

    # check model
    if os.access(MODEL_PATH, os.F_OK):
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        # load model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('=> no model.pth found')
        return -1

    model.to(device)
    # evaluate
    test(test_dataloader, model, device)


if __name__ == "__main__":
    main()
