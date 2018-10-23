import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
from libtiff import TIFF


MODEL_PATH = './model.pth'
IMAGE_PATH = './retinal_optic/Original_Images/'
GROUNDTRUTH_PATH = './retinal_optic/Groundtruths/'
TRAIN_NAME = 'Training'
TEST_NAME = 'Testing'
IMAGE_FORMAT = 'jpg'
GROUNDTRUTH_FORMAT = 'tif'


class RetinalSegDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_raw = cv2.imread(self.data_list[idx][0], cv2.IMREAD_UNCHANGED)
        # numpy image: H x W x C
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, None, fx=0.25, fy=0.25)
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        groundtruth_raw = TIFF.open(train_list[idx][1], mode='r')
        groundtruth = groundtruth_raw.read_image()
        # width * height （* channel_num）
        groundtruth = cv2.resize(groundtruth, (15, 10))
        groundtruth = np.reshape(groundtruth, (-1))
        groundtruth = torch.from_numpy(groundtruth)
        groundtruth = groundtruth.float()
        groundtruth = torch.where(
            groundtruth > 0,
            torch.ones(groundtruth.size()),
            torch.zeros(groundtruth.size()))
        sample = {'image': image, 'label': groundtruth}
        return sample


IMAGE_train_list = []
GROUNDTRUTH_train_list = []
IMAGE_train_glob = os.path.join(IMAGE_PATH, TRAIN_NAME, '*.' + IMAGE_FORMAT)
# 获取文件夹下所有匹配文件路径
IMAGE_train_list.extend(glob.glob(IMAGE_train_glob))
GROUNDTRUTH_train_glob = os.path.join(
    GROUNDTRUTH_PATH, TRAIN_NAME, '*.' + GROUNDTRUTH_FORMAT)
# 获取文件夹下所有匹配文件路径
GROUNDTRUTH_train_list.extend(glob.glob(GROUNDTRUTH_train_glob))

IMAGE_test_list = []
GROUNDTRUTH_test_list = []
IMAGE_test_glob = os.path.join(IMAGE_PATH, TEST_NAME, '*.' + IMAGE_FORMAT)
IMAGE_test_list.extend(glob.glob(IMAGE_test_glob))
GROUNDTRUTH_test_glob = os.path.join(
    GROUNDTRUTH_PATH, TEST_NAME, '*.' + GROUNDTRUTH_FORMAT)
GROUNDTRUTH_test_list.extend(glob.glob(GROUNDTRUTH_test_glob))

train_list = []
for i in range(len(IMAGE_train_list)):
    train_list.append([IMAGE_train_list[i], GROUNDTRUTH_train_list[i]])

test_list = []
for i in range(len(IMAGE_test_list)):
    test_list.append([IMAGE_test_list[i], GROUNDTRUTH_test_list[i]])

train_data = RetinalSegDataset(train_list)
test_data = RetinalSegDataset(test_list)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(128 * 22 * 33, 10 * 15)
        self.fc2 = nn.Linear(10 * 15, 10 * 15)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool1(x)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool2(x)
        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool3(x)
        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool4(x)
        x = F.relu(self.conv5(x), inplace=True)
        x = self.pool5(x)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()
epoch_saved = 0
# check model
if os.access(MODEL_PATH, os.F_OK):
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    # load model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    # necessary!
    model.eval()
    epoch_saved = checkpoint['epoch']
    print('————Model Loaded————')
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(model, epoch):
    for batch_idx, batched_sample in enumerate(train_loader):
        optimizer.zero_grad()
        image = batched_sample['image']
        image = image.to(device)
        label = batched_sample['label']
        label = label.to(device)
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(image), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, MODEL_PATH)  # save model


def test(model, epoch):
    loss_per_epoch = 0
    optimizer.zero_grad()
    for batch_idx, batched_sample in enumerate(test_loader):
        image = batched_sample['image']
        image = image.to(device)
        label = batched_sample['label']
        label = label.to(device)
        output = model(image)
        # reduce memory usage!!
        loss_per_epoch += float(criterion(output, label))
    loss_mean = loss_per_epoch / len(test_loader.dataset)
    print('Train Epoch: {} loss on test_data({} samples) is {:.6f}'.format(
        epoch, len(test_loader.dataset), loss_mean))


# def main():
for epoch in range(1, 5):
    train(model, epoch + epoch_saved)
    test(model, epoch + epoch_saved)
print('****************train complete******************')
for batch_idx, batched_sample in enumerate(test_loader):
    image = batched_sample['image']
    image = image.to(device)
    output = model(image)
    output = output[0]
    output = output.view(10, 15)
    output = output.to("cpu")
    print(output)
    output = torch.where(output > 0.5, torch.ones(
        output.size()), torch.zeros(output.size()))
    output = output.byte()
    label_test = output.numpy()
    tif = TIFF.open('./test/{}.tif'.format(55 + batch_idx), mode='w')
    tif.write_image(label_test)

# if __name__ == '__main__':
#    main()
