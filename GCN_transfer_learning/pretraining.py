import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import sklearn.metrics
import scipy.sparse

import coarsening

import multiprocessing
multiprocessing.set_start_method('spawn', True)


class Graph_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rescaled_Laplacian):
        super(Graph_Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel = nn.Linear(in_channels * kernel_size, out_channels)
        self.laplacian = nn.Parameter(rescaled_Laplacian.to_sparse(), requires_grad=False)

    def forward(self, input):
        """ input size: (Batch, Channel, Vertex)
        """
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

        x = x.view([self.kernel_size, V, C, B])         # K, V, C, B
        x = x.permute([3, 1, 2, 0]).contiguous()        # B, V, C, K
        x = x.view([B*V, C*self.kernel_size])           # B*V, C*K

        x = self.kernel(x)                              # B*V, C_out
        x = x.view([B, V, self.out_channels])           # B, V, C_out
        x = x.permute([0, 2, 1]).contiguous()           # B, C_out, V

        return x


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

        # Sparse Layer
        sparse = torch.zeros((IN_W//2, IN_W//2), dtype=torch.float32)
        self.sparse = nn.Parameter(sparse, requires_grad=True)
        self.sparse_dim = CL1_F

    def forward(self, x, prob=0):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Sparse Layer
        sparse_3 = self.sparse.unsqueeze(0)
        m, n = self.sparse.shape
        sparse_expand = sparse_3.expand((self.sparse_dim, m, n))
        sparse_sigmoid = torch.sigmoid(sparse_expand)
        x = x * sparse_sigmoid

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


class Net_GCN_mnist(nn.Module):
    def __init__(self, net_parameters):
        super(Net_GCN_mnist, self).__init__()

        # network parameters
        IN_W, IN_H, IN_C, CL1_F, CL1_K, IN_V, CL2_F, CL2_K, CL3_F, CL3_K, FC1_F, FC2_F, L, node_index, perm = net_parameters
        FC1_IN = CL3_F * (IN_V // 16)
        self.node_index = node_index
        self.perm = perm

        # Convolutional Layer
        PAD1 = (CL1_K - 1) // 2
        self.conv1 = nn.Conv2d(IN_C, CL1_F, CL1_K, padding=PAD1)
        Fin = IN_C * CL1_K**2
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.weight.data.uniform_(-scale, scale)
        self.conv1.bias.data.fill_(0.0)

        # Batch Normalization Layer
        self.norm = nn.BatchNorm1d(CL1_F, affine=False)

        # Graph Convolutional Layer 1
        self.conv2 = Graph_Conv(CL1_F, CL2_F, CL2_K, L[0])
        Fin = CL1_F * CL2_K
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.kernel.weight.data.uniform_(-scale, scale)
        self.conv2.kernel.bias.data.fill_(0.0)

        # Graph Convolutional Layer 2
        self.conv3 = Graph_Conv(CL2_F, CL3_F, CL3_K, L[2])
        Fin = CL2_F * CL3_K
        Fout = CL3_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv3.kernel.weight.data.uniform_(-scale, scale)
        self.conv3.kernel.bias.data.fill_(0.0)

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

        # Graph Max Pooling
        self.gpool = Graph_MaxPool(4)

        # Sparse Layer
        sparse = torch.zeros((IN_W//2, IN_W//2), dtype=torch.float32)
        self.sparse = nn.Parameter(sparse, requires_grad=True)
        self.sparse_dim = CL1_F

    def forward(self, x, prob=0):
        # Convolutional Layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Grid to Graph
        x = x[:, :, self.node_index[:,0], self.node_index[:,1]]

        # Permute Nodes
        B, C, V = x.shape
        V_perm = len(self.perm)
        x_append = torch.zeros([B, C, V_perm - V], dtype=x.dtype, device=x.device)
        x = torch.cat((x, x_append), dim=2)
        x = x[:, :, self.perm]

        # Batch Normalization Layer
        x = self.norm(x)

        # Graph Convolutional Layer 1
        x = self.conv2(x)
        x = F.relu(x)
        x = self.gpool(x)

        # Graph Convolutional Layer 2
        x = self.conv3(x)
        x = F.relu(x)
        x = self.gpool(x)

        # Full Connected Layer 1
        x = x.view(-1, self.FC1_IN)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=prob, training=self.training)

        # Full Connected Layer 2
        x = self.fc2(x)

        return x


def train(args, model, device, train_loader, optimizer, epoch, stat="CCN"):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward
        output = model(data, prob=args.drop)
        loss_L1 = torch.norm(torch.sigmoid(model.sparse), p=1)
        loss_var = torch.var(torch.sigmoid(model.sparse).view(-1,))
        if stat == "CCN":
            sparse_rate = 0.005
            var_rate = 0.01
        else:
            sparse_rate = 0
            var_rate = 0
        loss = F.cross_entropy(output, target) + sparse_rate * loss_L1 + var_rate / (0.01 + loss_var)

        # Backword
        loss.backward()

        # Update
        optimizer.step()

        # Evaluate
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    if stat == "CCN":
        w = torch.sigmoid(model.sparse.cpu()).data.numpy()
        df = pd.DataFrame(w)
        df.to_csv("D:/code/DTI_data/output/epoch_{}.csv".format(epoch), header=False, index=False)


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


def node_select(PATH):
    df = pd.read_csv(PATH, header=None)
    activate = df.to_numpy()
    ii = np.unravel_index(np.argsort(activate.ravel())[-90:], activate.shape)    # get the indices of top 90
    iix = np.expand_dims(ii[0], axis=1)
    iiy = np.expand_dims(ii[1], axis=1)
    iixy = np.concatenate((iix, iiy), axis=1)

    groupgraph = "D:/code/DTI_data/network_distance/grouplevel.edge"
    ggraph = pd.read_csv(groupgraph, sep='\t', header=None).to_numpy()

    dist = sklearn.metrics.pairwise_distances(iixy, metric="euclidean")
    dist_exp = np.exp(-dist**2 / (2 * 1.5**2))

    ggraph_ev, _ = np.linalg.eig(ggraph)
    dist_ev, _ = np.linalg.eig(dist_exp)

    gindex = np.argsort(ggraph_ev)
    dindex = np.argsort(dist_ev)

    sort_index = np.zeros((90, 2))
    sort_index[gindex] = iixy[dindex]

    return sort_index


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST")
    parser.add_argument('-B', '--batchsize', type=int, default=1, metavar='B')
    parser.add_argument('-E', '--epochs', type=int, default=30, metavar='N')
    parser.add_argument('-L', '--lr', type=float, default=0.01, metavar='LR')
    parser.add_argument('-MT', '--momentum', type=float, default=0.9, metavar='MT')
    parser.add_argument('-D', '--drop', type=float, default=0.3, metavar='D')
    parser.add_argument('-S', '--seed', type=int, default=1, metavar='S')
    parser.add_argument('-C', '--cuda', action='store_true', default=False)
    parser.add_argument('-GG', '--groupgraph', default='D:/code/DTI_data/network_distance/grouplevel.edge')
    parser.add_argument('-M', '--model', default='model.pth', metavar='PATH', help='path to model')
    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
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
    """
    # ----------------- CCN part ----------------- #
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

    """
    # ----------------- GCN part ----------------- #
    PATH = "D:/code/DTI_data/pretrain/191105_sr_0.005_var/epoch_100.csv"
    node_index = node_select(PATH)

    # group-level graph
    ggraph = pd.read_csv(args.groupgraph, sep='\t', header=None).to_numpy()
    ggraph = scipy.sparse.csr_matrix(ggraph)

    # graph coarsening
    L, perm = coarsening.coarsen(ggraph, 4)
    r_L_torch = []      # list of rescaled Ls for pytorch processing
    for l in L:
        r_L_torch.append(torch.from_numpy(coarsening.rescaled_L(l).toarray()).float())

    # GCN network parameters
    IN_W = 28
    IN_H = 28
    IN_C = 1
    CL1_F = 3
    CL1_K = 5
    IN_V = len(perm)
    CL2_F = 16
    CL2_K = 8
    CL3_F = 32
    CL3_K = 8
    FC1_F = 50
    FC2_F = 10
    net_GCN_parameters = [IN_W, IN_H, IN_C, CL1_F, CL1_K, IN_V, CL2_F, CL2_K, CL3_F, CL3_K, FC1_F, FC2_F, r_L_torch, node_index, perm]
    model = Net_GCN_mnist(net_GCN_parameters).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, stat="GCN")
        test(args, model, device, test_loader)

    GCL1_w = model.conv2.kernel.weight.data
    GCL1_b = model.conv2.kernel.bias.data
    GCL2_w = model.conv3.kernel.weight.data
    GCL2_b = model.conv3.kernel.bias.data
    FC1_w = model.fc1.weight.data
    FC1_b = model.fc1.bias.data

    torch.save({
        'GCL1_w': GCL1_w,
        'GCL1_b': GCL1_b,
        'GCL2_w': GCL2_w,
        'GCL2_b': GCL2_b,
        'FC1_w': FC1_w,
        'FC1_b': FC1_b
    }, args.model)


if __name__ == "__main__":
    main()
