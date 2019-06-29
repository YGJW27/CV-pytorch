import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
import pandas as pd

from graph import construction, coarsening


class Graph_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, laplacian):
        super(Graph_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = nn.Linear(in_channels*kernel_size, out_channels)
        self.lambda_max = self.Lambda_max(laplacian)
        self.laplacian = nn.Parameter(self.scaled_Laplacian(laplacian).to_sparse(), requires_grad=False)

    def forward(self, input):
        """ input size: (Batch, Channel, Vertex)
        """
        assert input.is_contiguous()

        B, C, V = input.shape
        input = input.permute([2, 1, 0]).contiguous()   # V, C, B
        x0 = input.view([V, C*B])                       # V, C*B
        x = x0.unsqueeze(0)                             # 1, V, C*B

        if self.kernel_size > 1:
            x1 = torch.mm(self.laplacian, x0)           # V, C*B
            x = torch.cat((x, x1.unsqueeze(0)), dim=0)  # 2, V, C*B
        for k in range(2, self.kernel_size):
            x_k = 2 * torch.mm(self.laplacian, x[k-1]) - x[k-2]     # V, C*B
            x = torch.cat((x, x_k.unsqueeze(0)), dim=0)             # k+1, V, C*B
        
        assert x.shape[0] == self.kernel_size
        x = x.view([self.kernel_size, V, C, B])         # K, V, C, B
        x = x.permute([3, 1, 2, 0]).contiguous()        # B, V, C, K
        x = x.view([B*V, C*self.kernel_size])           # B*V, C*K

        x = self.kernel(x)                              # B*V, C_out
        x = x.view([B, V, self.out_channels])           # B, V, C_out
        x = x.permute([0, 2, 1]).contiguous()           # B, C_out, V

        return x

    def Lambda_max(self, laplacian):
        return torch.eig(laplacian)[0][:, 0].max()

    def scaled_Laplacian(self, laplacian):
        M, _ = laplacian.shape
        laplacian = 2*laplacian / self.lambda_max - torch.ones(M, dtype=laplacian.dtype).diag()
        return laplacian


class Graph_MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(Graph_MaxPool, self).__init__()
        self.kernel = nn.MaxPool1d(kernel_size, stride=kernel_size)

    def forward(self, input):
        x = self.kernel(input)
        return x


class Net(nn.Module):
    def __init__(self, net_parameters):
        super(Net, self).__init__()

        # network parameters
        IN_C, IN_V, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, L = net_parameters
        FC1_IN = CL2_F * IN_V // 16

        # Graph Convolutional Layer 1
        self.conv1 = Graph_Conv(IN_C, CL1_F, CL1_K, L[0])
        Fin = IN_C * CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.kernel.weight.data.uniform_(-scale, scale)
        self.conv1.kernel.bias.data.fill_(0.0)

        # Graph Convolutional Layer 2
        self.conv2 = Graph_Conv(CL1_F, CL2_F, CL2_K, L[2])
        Fin = CL1_F * CL2_K
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.kernel.weight.data.uniform_(-scale, scale)
        self.conv2.kernel.bias.data.fill_(0.0)

        # Full Connected Layer 1
        self.fc1 = nn.Linear(FC1_IN, FC1_F)
        Fin = FC1_IN
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)

        # Full Connected Layer 2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # Max Pooling
        self.pool = Graph_MaxPool(4)

    def forward(self, x, prob=0):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Full Connected Layer 1
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=prob, training=self.training)

        # Full Connected Layer 2
        x = self.fc2(x)

        return x


class MRI_Dataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, idx):
        filepath, target = self.data_list[idx]
        dataframe = pd.read_csv(filepath, sep=" ", header=None)
        w = dataframe.to_numpy()
        w = construction.to_graph_signal(w)
        return w, target

    def __len__(self):
        return len(self.data_list)


class perm_data(object):
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, pic):
        C, V = pic.shape
        V_perm = len(self.indices)
        pic_append = torch.zeros([C, V_perm - V], dtype=pic.dtype)
        pic_new = torch.cat((pic, pic_append), dim=1)
        pic_new_perm = pic_new[:, self.indices]
        return pic_new_perm



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
    parser.add_argument('-D', '--drop', type=float, default=0.5, metavar='D')
    parser.add_argument('-S', '--seed', type=int, default=1, metavar='S')
    parser.add_argument('-C', '--cuda', action='store_true', default=False)
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # distance graph construction
    PATH = "D:/code/DTI_data/network_distance/AAL_116.node"
    theta = 30
    k = 50
    dist_A = construction.dist_graph(PATH, theta, k)

    # graph coarsening
    L, perm = coarsening.coarsen(dist_A, 4)
    L_torch = []
    for l in L:
        L_torch.append(torch.from_numpy(l.toarray()))

    # graph signal input construction


    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           perm_data(perm)
                       ])),
        batch_size=args.batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           perm_data(perm)
                       ])),
        batch_size=args.batchsize, shuffle=True, **kwargs)

    # network parameters

    IN_C, IN_V = train_loader.dataset[0][0].shape
    CL1_F = 32
    CL1_K = 25
    CL2_F = 64
    CL2_K = 25
    FC1_F = 512
    FC2_F = 10
    net_parameters = [IN_C, IN_V, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, L_torch]
    model = Net(net_parameters).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()