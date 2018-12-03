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
IMAGE_PATH = './dataset/'
IMAGE_FORMATS = ['jpg', 'jpeg', 'JPG', 'JPEG']  # windows下不区分大小写，ubuntu下区分
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def split_dataset():
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
        image = cv2.resize(image, (1366, 1024))     # resize: width * height
        image = image.transpose((2, 0, 1))  # numpy image -> torch image
        image = torch.from_numpy(image).float()
        image = image / 255.

        sample = {'image': image,
                  'label': self.data_list[idx][1],
                  'filename': self.data_list[idx][0]}
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


def train(train_dataloader, model, criterion, optimizer, epoch, device):
    # switch to train mode
    model.train()

    train_loss = 0
    for batch_idx, batched_sample in enumerate(train_dataloader):
        inputs = batched_sample['image']
        inputs = inputs.to(device)
        labels = batched_sample['label']
        labels = labels.to(device)

        output = model(inputs)
        loss = criterion(output, labels)
        train_loss += loss.item() * inputs.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_dataloader.dataset),
                100. * batch_idx * len(inputs)/ len(train_dataloader.dataset), loss))

    train_loss = train_loss / len(train_dataloader.dataset)
    print('train loss is:{:.6f}'.format(train_loss))


def validate(valid_dataloader, model, criterion, device):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, batched_sample in enumerate(valid_dataloader):
            inputs = batched_sample['image']
            inputs = inputs.to(device)
            labels = batched_sample['label']
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation images: {}%'.format(
        100 * correct/total))


def save_checkpoint(state, filename=MODEL_PATH):
    torch.save(state, filename)


def main():
    if os.access('dataset_list.pth', os.F_OK):
        print('=> loading dataset_list')
        data_point = torch.load('dataset_list.pth', map_location='cpu')

        train_list = data_point['train_list']
        valid_list = data_point['valid_list']
        test_list = data_point['test_list']
    else:
        print('=> no dataset_list found')
        [train_list, valid_list, test_list] = split_dataset()

        torch.save({
            'train_list': train_list,
            'valid_list': valid_list,
            'test_list': test_list
        }, 'dataset_list.pth')

    train_dataset = Classify_Dataset(train_list)
    valid_dataset = Classify_Dataset(valid_list)
    test_dataset = Classify_Dataset(test_list)

    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net()

    # check model
    if os.access(MODEL_PATH, os.F_OK):
        print('=> loading checkpoint')
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        # load model parameters
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print('=> no checkpoint found')
        start_epoch = 0

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(start_epoch, 200):
        # train and evaluate
        train(train_dataloader, model, criterion, optimizer, epoch, device)
        validate(valid_dataloader, model, criterion, device)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict()
        })


if __name__ == "__main__":
    main()
