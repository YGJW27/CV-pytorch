import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self, net_parameters):
        super(Net, self).__init__()

        # network parameters
        IN_W, IN_H, IN_C, CL1_F, CL1_K, CL2_F, CL2_K, CL3_F, CL3_K, FC1_F, FC2_F = net_parameters
        FC1_IN = CL3_F * (IN_W // 4)**2

        # Convolutional Layer 1
        PAD1 = (CL1_K - 1) // 2
        self.conv1 = nn.Conv2d(IN_C, CL1_F, CL1_K, padding=PAD1)
        Fin = IN_C * CL1_K**2
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.weight.data.uniform_(-scale, scale)
        self.conv1.bias.data.fill_(0.0)

        # Convolutional Layer 2
        PAD2 = (CL2_K - 1) // 2
        self.conv2 = nn.Conv2d(CL1_F, CL2_F, CL2_K, padding=PAD2)
        Fin = CL1_F * CL2_K**2
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.weight.data.uniform_(-scale, scale)
        self.conv2.bias.data.fill_(0.0)

        # Convolutional Layer 3
        PAD3 = (CL3_K - 1) // 2
        self.conv3 = nn.Conv2d(CL2_F, CL3_F, CL3_K, padding=PAD3)
        Fin = CL2_F * CL3_K**2
        Fout = CL3_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv3.weight.data.uniform_(-scale, scale)
        self.conv3.bias.data.fill_(0.0)

        # Full Connected Layer 1
        self.fc1 = nn.Linear(FC1_IN, FC1_F)
        Fin = FC1_IN
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1_IN = FC1_IN

        # Full Connected Layer 2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, prob=0):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # Full Connected Layer 1
        x = x.view(-1, self.FC1_IN)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=prob, training=self.training)

        # Full Connected Layer 2
        x = self.fc2(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward
        output = model(data, prob=args.drop)
        loss = F.cross_entropy(output, target)

        # Backword
        loss.backward()

        # Update
        optimizer.step()

        # Evaluate
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.4f}%\n'.format(
        test_loss, 100. * accuracy))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST")
    parser.add_argument('-B', '--batchsize', type=int, default=1, metavar='B')
    parser.add_argument('-E', '--epochs', type=int, default=30, metavar='N')
    parser.add_argument('-L', '--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('-M', '--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('-D', '--drop', type=float, default=0, metavar='D')
    parser.add_argument('-S', '--seed', type=int, default=1, metavar='S')
    parser.add_argument('-C', '--cuda', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batchsize, shuffle=True, **kwargs)

    # network parameters
    IN_W = 28
    IN_H = 28
    IN_C = 1
    CL1_F = 3
    CL1_K = 5
    CL2_F = 16
    CL2_K = 5
    CL3_F = 32
    CL3_K = 5
    FC1_F = 100
    FC2_F = 10
    net_parameters = [IN_W, IN_H, IN_C, CL1_F, CL1_K, CL2_F, CL2_K, CL3_F, CL3_K, FC1_F, FC2_F]
    model = Net(net_parameters).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
