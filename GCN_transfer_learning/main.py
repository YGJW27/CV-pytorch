import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.model_selection import KFold

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
    def __init__(self, net_parameters, drop, pretrain_params):
        super(Net, self).__init__()

        # network parameters
        IN_C, IN_V, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, L = net_parameters
        FC1_IN = CL2_F * IN_V // 16
        self.drop = drop

        # Batch Normalizaton Layer
        self.norm = nn.BatchNorm1d(IN_C, affine=False)

        # Graph Convolutional Layer 1
        self.conv1 = Graph_Conv(IN_C, CL1_F, CL1_K, L[0])
        Fin = IN_C * CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv1.kernel.weight.data = pretrain_params['GCL1_w']
        self.conv1.kernel.bias.data = pretrain_params['GCL1_b']
        self.conv1.kernel.weight.requires_grad_(False)
        self.conv1.kernel.bias.requires_grad_(False)

        # Graph Convolutional Layer 2
        self.conv2 = Graph_Conv(CL1_F, CL2_F, CL2_K, L[2])
        Fin = CL1_F * CL2_K
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.conv2.kernel.weight.data = pretrain_params['GCL2_w']
        self.conv2.kernel.bias.data = pretrain_params['GCL2_b']
        self.conv2.kernel.weight.requires_grad_(False)
        self.conv2.kernel.bias.requires_grad_(False)

        # Full Connected Layer 1
        self.fc1 = nn.Linear(FC1_IN, FC1_F)
        Fin = FC1_IN
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc1.weight.data = pretrain_params['FC1_w']
        self.fc1.bias.data = pretrain_params['FC1_b']
        self.fc1.weight.requires_grad_(True)
        self.fc1.bias.requires_grad_(True)

        # Full Connected Layer 2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = np.sqrt(2.0 / (Fin + Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # Max Pooling
        self.pool = Graph_MaxPool(4)

    def forward(self, x):
        # Batch Normalization Layer
        x = self.norm(x)

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
        x = torch.sigmoid(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        # Full Connected Layer 2
        x = self.fc2(x)

        return x


def data_list(sample_path):
    sub_dirs = [x[0] for x in os.walk(sample_path)]
    sub_dirs.pop(0)

    data_list = []

    for sub_dir in sub_dirs:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        file_glob = os.path.join(sample_path, dir_name, '*')
        file_list.extend(glob.glob(file_glob))

        for file_name in file_list:
            data_list.append([file_name, dir_name])

    return np.array(data_list)


class MRI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, idx):
        filepath, target = self.data_list[idx][0], int(self.data_list[idx][1])
        dataframe = pd.read_csv(filepath, sep="\t", header=None)
        pic = dataframe.to_numpy()          # (V, C)
        pic = np.transpose(pic, (1, 0))     # (C, V)

        if self.transform is not None:
            pic = self.transform(pic)

        return pic, target

    def __len__(self):
        return len(self.data_list)


class Array_To_Tensor(object):
    def __call__(self, input):
        return torch.from_numpy(input).float()


class Perm_Data(object):
    def __init__(self, indices):
        self.indices = indices

    def __call__(self, pic):
        C, V = pic.shape
        V_perm = len(self.indices)
        pic_append = torch.zeros([C, V_perm - V], dtype=pic.dtype)
        pic_new = torch.cat((pic, pic_append), dim=1)
        pic_new_perm = pic_new[:, self.indices]
        return pic_new_perm


def train(model, device, train_loader, optimizer, weight_decay, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward
        output = model(data)
        cls_loss = F.cross_entropy(output, target)
        reg_loss = 0
        for name, param in model.named_parameters():
            if ('weight' in name) and ('fc' in name):
                reg_loss += torch.norm(param)

        loss = cls_loss + weight_decay * reg_loss

        # Backword
        loss.backward()

        # Update
        optimizer.step()

        # Evaluate
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
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

    return accuracy, test_loss


def cross_validate(args, dataset, cv, lr, w_d, mmt, drop, perm, net_parameters):
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    acc_sum = 0
    kf = KFold(n_splits=cv, shuffle=True, random_state=args.dataseed)
    for idx, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        torch.manual_seed(args.modelseed)
        torch.cuda.manual_seed_all(args.modelseed)

        if os.access(args.model, os.F_OK):
            pretrain_params = torch.load(args.model, map_location='cpu')
            print('—————Model Loaded—————')
        else:
            print('————Loading Failed————')

        model = Net(net_parameters, drop, pretrain_params)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mmt)
        print('--------Cross Validation: {}/{} --------\n'.format(idx + 1, cv))
        kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
        train_loader = torch.utils.data.DataLoader(
            MRI_Dataset(dataset[train_idx],
                        transform=transforms.Compose([
                            Array_To_Tensor(),
                            Perm_Data(perm)
                        ])),
            batch_size=args.batchsize, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            MRI_Dataset(dataset[test_idx],
                        transform=transforms.Compose([
                            Array_To_Tensor(),
                            Perm_Data(perm)
                        ])),
            batch_size=args.batchsize, shuffle=True, **kwargs)
        loss_list = []
        filename = 'D:/code/DTI_data/result/' + "lr" + str(lr) \
            + "_wd" + str(w_d) + "_mmt" + str(mmt) + "_drop" \
            + str(drop) + "_cv" + str(idx) + ".csv"

        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, w_d, epoch)
            accuracy, loss = test(model, device, test_loader)
            loss_list.append(loss)
            if epoch > 200:
                if (loss_list[-101] - loss_list[-1]) < 1e-5:
                    break
        loss_df = pd.DataFrame(loss_list)
        loss_df.to_csv(filename, header=False, index=False)
        acc_sum += accuracy
    return acc_sum / cv, loss, epoch


def main():
    parser = argparse.ArgumentParser(description="Pytorch MNIST")
    parser.add_argument('-B', '--batchsize', type=int, default=20, metavar='B')
    parser.add_argument('-E', '--epochs', type=int, default=5000, metavar='N')
    parser.add_argument('-C', '--cuda', action='store_true', default=False)
    parser.add_argument('-DS', '--dataseed', type=int, default=2, metavar='S')
    parser.add_argument('-MS', '--modelseed', type=int, default=2, metavar='S')
    parser.add_argument('-GG', '--groupgraph', default='D:/code/DTI_data/network_distance/grouplevel.edge')
    parser.add_argument('-DP', '--datapath', default='D:/code/DTI_data/output/')
    parser.add_argument('-IC', '--channel', type=int, default=3, metavar='N')
    parser.add_argument('-M', '--model', default='model.pth', metavar='PATH', help='path to model')
    args = parser.parse_args()

    # group-level graph
    ggraph = pd.read_csv(args.groupgraph, sep='\t', header=None).to_numpy()
    ggraph = scipy.sparse.csr_matrix(ggraph)

    # graph coarsening
    L, perm = coarsening.coarsen(ggraph, 4)
    r_L_torch = []      # list of rescaled Ls for pytorch processing
    for l in L:
        r_L_torch.append(torch.from_numpy(coarsening.rescaled_L(l).toarray()).float())

    # net parameters
    IN_C = args.channel
    IN_V = len(perm)
    CL1_F = 16
    CL1_K = 8
    CL2_F = 32
    CL2_K = 8
    FC1_F = 50
    FC2_F = 2
    net_parameters = [IN_C, IN_V, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F, r_L_torch]

    dataset = data_list(args.datapath)

    # 10-fold cross validation dataset split
    # lr_array = [1e-2, 4e-2, 7e-2, 1e-3, 4e-3, 7e-3, 1e-4]
    # weight_decay = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # momentum = [0.85, 0.9, 0.95]
    # drop_array = [0.2, 0.3, 0.4, 0.5]
    lr_array = [1e-3]
    weight_decay = [0.01]
    momentum = [0.85]
    drop_array = [0.3]

    result_path = 'D:/code/DTI_data/result/cross_validation.csv'
    df = pd.DataFrame(columns=['learn_rate', 'weight_decay', 'momentum',
                      'drop_rate', 'accuracy', 'loss', 'epoch'])
    df.to_csv(result_path, header=True, index=False)

    for lr in lr_array:
        for w_d in weight_decay:
            for mmt in momentum:
                for drop in drop_array:
                    print("lr: ", lr, "w_d: ", w_d, "momentum: ", mmt, "drop: ", drop)
                    acc, loss, epoch = cross_validate(args, dataset, 20, lr, w_d, mmt, drop, perm, net_parameters)
                    print("lr: ", lr, "w_d:  ", w_d, "momentum: ", mmt, "drop: ", drop, "acc: ", acc)
                    df = pd.read_csv(result_path, header=0)
                    df = df.append({'learn_rate': lr, 'weight_decay': w_d,
                                    'momentum': mmt, 'drop_rate': drop, 'accuracy': acc,
                                    'loss': loss, 'epoch': epoch}, ignore_index=True)
                    df.to_csv(result_path, header=True, index=False)


if __name__ == "__main__":
    main()
