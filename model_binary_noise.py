import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import get_laplacian
import math
import argparse
import os.path as osp
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

### For Binary noise

def nodeAttack(x, device, ratio, normal=False):
    if normal:
            x_new = x + torch.normal(0, ratio, size=(x.shape[0], x.shape[1])).to(device)
            x_new -= x_new.min() # make sure all values are non-negative
    else:
            mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
            mask = mask.to(device)
            x -= mask.int()
            x_new = (torch.abs(x)==1).double()
    return x_new.to(device)

def rigrsure(x, N1, N2, col_idx):
    """
    Adaptive threshold selection using principle of Stein's Unbiased Risk Estimate (SURE).

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param N1: torch dense tensor with shape [num_nodes, num_hid_features]
    :param N2: torch dense tensor with shape [num_nodes, num_hid_features]
    :param col_idx: torch dense tensor with shape [num_hid_features]
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape

    sx, _ = torch.sort(torch.abs(x), dim=0)
    sx2 = sx ** 2
    CS1 = torch.cumsum(sx2, dim=0)
    risks = (N1 + CS1 + N2 * sx2) / n
    best = torch.argmin(risks, dim=0)
    thr = sx[best, col_idx]

    return thr


def multiScales(x, r, Lev, num_nodes):
    """
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range(Lev, r * Lev):
        if block_idx == Lev:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev, 'something wrong in multiScales'
    return specEnergy


def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr


def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    assert mode in ('soft', 'hard'), 'shrinkage type is invalid'

    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)


# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c

# function for pre-processing
# Please refer to the "corrected" version get_operator2
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d

def get_operator2(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(Lev):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J - l) / a) * L) @ T0F - T0F
            d[j, l] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J - l)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l] += c[j][k] * TkF
        FD1 = d[0, l]

    return d

# function for pre-processing For no Chebyshev approximation: Slow
def get_operator1(L, DFilters, lambdas, eigenvecs, s, Lev):
    lambdas[lambdas <= 0.0] = 0.0
    lambdas[lambdas > 2.0] = 2.0
    r = len(DFilters)   # DFiliters is a list of g functions
    J =  np.log(lambdas[-1] / np.pi) / np.log(s)  # dilation level to start the decomposition
    #a = np.pi / 2  # consider the domain of masks as [0, pi]
    d = dict()
    FD1 = 1.0
    for l in range(Lev):
        for j in range(r):
            d[j, l] = FD1 * DFilters[j](s**(- J - l) * lambdas)
        FD1 = d[0, l]

    d_list = list()
    for i in range(r):
        for l in range(Lev):
            print('Calculating Matrix...{0:2d}, {1:2d}'.format(i, l))
            d_list.append(np.matmul(eigenvecs, np.diag(d[i, l]) @ eigenvecs.T))

    print(FD1)
    return d_list

class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage, sigma, bias=True, activation = F.elu, cutoff = 'NoCutoff'):
        super(UFGConv, self).__init__()
        self.r = r
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.sigma = sigma
        self.cutoff = cutoff
        self.crop_len = (Lev - 1) * num_nodes
        self.act = activation
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
            self.cutoff_idx = torch.ones(self.r * self.Lev * self.num_nodes, 1).cuda()
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
            self.cutoff_idx = torch.ones(self.r * self.Lev * self.num_nodes, 1)
        if self.cutoff == 'PartialCutoff':
            self.cutoff_idx[-self.Lev * self.num_nodes:-(self.Lev - 1) * self.num_nodes] = 0.0  # Cutting off g3's lowest level (W_{K,0})
        if self.cutoff == 'FullCutoff':
            self.cutoff_idx[-self.Lev * self.num_nodes:] = 0.0  # Cutting off entire g3's levels (W_{K,l}) best??
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        # x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        x = torch.matmul(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Hadamard product in spectral domain
        #if self.cutoff:
        #    if torch.cuda.is_available():
        #        cutoff = torch.ones(self.r * self.Lev * self.num_nodes, 1).to(x.get_device())
        #    else:
        #        cutoff = torch.ones(self.r * self.Lev * self.num_nodes, 1)
        #    # cutoff[-self.Lev * self.num_nodes:-(self.Lev - 1) * self.num_nodes] = 0.0     # Cutting off g3's lowest level (W_{K,0})
        #    cutoff[-self.Lev * self.num_nodes:] = 0.0  # Cutting off entire g3's levels (W_{K,l}) best??
        #    x = (cutoff * self.filter) * x
        #else:
        #    x = self.filter * x
        x = (self.cutoff_idx * self.filter) * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # calculate the scales for thresholding
        ms = multiScales(x, self.r, self.Lev, self.num_nodes)

        # perform wavelet shrinkage
        for block_idx in range(self.Lev - 1, self.r * self.Lev):
            ms_idx = 0
            if block_idx == self.Lev - 1:  # low frequency block
                x_shrink = x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :]
            else:  # remaining high frequency blocks with wavelet shrinkage
                x_shrink = torch.cat((x_shrink,
                                      waveletShrinkage(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                       simpleLambda(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                                    ms[ms_idx], self.sigma), mode=self.shrinkage)), dim=0)
                ms_idx += 1

        # Fast Tight Frame Reconstruction
        # x_shrink = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x_shrink)
        x_shrink = torch.matmul(torch.cat(d_list[self.Lev - 1:], dim=1), x_shrink)

        if self.bias is not None:
            x_shrink += self.bias
        if self.act is not None:
            x_shrink = self.act(x_shrink)
        return x_shrink


class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, r, Lev, num_nodes, shrinkage='soft', sigma=1.0, dropout_prob=0.5, activation = None, cutoff1 = 'NoCutoff', cutoff2 = 'NoCutoff'):
        super(Net, self).__init__()
        self.GConv1 = UFGConv(num_features, nhid, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma, activation = activation, cutoff = cutoff1)   #it is bad with activation
        self.GConv2 = UFGConv(nhid, num_classes, r, Lev, num_nodes, shrinkage=shrinkage, sigma=sigma, activation = activation, cutoff = cutoff2)    #it is bad with activation
        self.drop1 = nn.Dropout(dropout_prob)

    def forward(self, data, d_list):
        x = data.x  # x has shape [num_nodes, num_input_features]

        x = self.GConv1(x, d_list)
        x = self.drop1(x)
        x = self.GConv2(x, d_list)

        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='name of dataset (default: Cora)')
    parser.add_argument('--reps', type=int, default=10,
                        help='number of repetitions (default: 10)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate (default: 5e-3)')
    parser.add_argument('--wd', type=float, default=0.01,
                        help='weight decay (default: 5e-3)')
    parser.add_argument('--nhid', type=int, default=16,
                        help='number of hidden units (default: 16)')
    parser.add_argument('--Lev', type=int, default=2,
                        help='level of transform (default: 2)')
    parser.add_argument('--s', type=float, default=2.5,
                        help='dilation scale > 1 (default: 2)')
    parser.add_argument('--n', type=int, default=2,
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    parser.add_argument('--FrameType', type=str, default='Entropy',
                        help='frame type (default: Entropy): Sigmoid, Entropy')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability (default: 0.3)')
    parser.add_argument('--shrinkage', type=str, default='soft',
                        help='soft or hard thresholding (default: soft)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='standard deviation of the noise (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha value in Framelet function (default: 0.5 for Entropy; 20.0 for Sigmoid)')   #newly added parameter
    parser.add_argument('--noiseLev', type=float, default=0.5,
                        help='Added noise level (default: 0.5)')           #newly added parameter
    parser.add_argument('--cutoff1', type=str, default='NoCutoff',
                        help='Whether to use cutoff of high frequency in first layer (default: false): NoCutoff, PartialCutoff, FullCutoff')  #newly added parameter  We only set this to True if we really want to test the way of cutting off g3
    parser.add_argument('--cutoff2', type=str, default='NoCutoff',
                        help='Whether to use cutoff of high frequency in second layer (default: false): NoCutoff, PartialCutoff, FullCutoff')  # newly added parameter  We only set this to True if we really want to test the way of cutting off g3
    parser.add_argument('--seed', type=int, default=1000,
                        help='random seed (default: 1000)')
    #parser.add_argument('--filename', type=str, default='results',
    #                    help='filename to store results and the model (default: results)')
    parser.add_argument('--ExpNum', type=int, default='1',
                        help='The Experiment Number (default: 1)')
    parser.add_argument('--Chebyshev', default=True, action='store_false',
                        help='Whether to use Chebyshev approximation (default: True)')
    parser.add_argument('--activation', default=False, action='store_true',
                        help='Whether to use Chebyshev approximation (default: True)')
    parser.add_argument('--FrequencyNum', type=int, default=100,
                        help='The number of (noise) high frequency components (default: 100)')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset and prepare noise through high frequence of Laplacian
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), 'data', dataname)
    dataset = Planetoid(root=rootname, name=dataname)

    num_nodes = dataset[0].x.shape[0]
    nfeatures = dataset[0].x.shape[1]
    L = get_laplacian(dataset[0].edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    lambdas, eigenvecs = np.linalg.eigh(L.todense())
    lambda_max = lambdas[-1]
    #u = eigenvecs[:, -args.FrequencyNum:]
    #w = args.noiseLev * np.random.rand(args.FrequencyNum, nfeatures)
    #w = np.random.normal(loc=0, scale=args.noiseLev, size=(args.FrequencyNum, nfeatures))
    #noises = torch.tensor(np.matmul(u, w)).to(device)
    data = dataset[0].to(device)
    #data.x = data.x + noises
    data.x = nodeAttack(data.x, device, args.noiseLev)


    FrameType = args.FrameType
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Sigmoid':
        alpha = args.alpha    # make sure default value = 20.0
        D1 = lambda x: np.sqrt(1.0 - 1.0 / (1.0+np.exp(-alpha*(x/np.pi-0.5))))
        D2 = lambda x: np.sqrt(1.0 / (1.0+np.exp(-alpha*(x/np.pi-0.5))))
        DFilters = [D1, D2]
        RFilters = [D1, D2]
    elif FrameType == 'Entropy':
        alpha = args.alpha   # with a default value = 0.5  (can be made a tunable parameter)
        D1 = lambda x: np.sqrt((1 - alpha*4*(x/np.pi) + alpha*4*(x/np.pi)*(x/np.pi))*((x/np.pi)<=0.5))
        D2 = lambda x: np.sqrt(alpha*4*(x/np.pi) - alpha*4*(x/np.pi)*(x/np.pi))
        D3 = lambda x: np.sqrt((1 - alpha*4*(x/np.pi) + alpha*4*(x/np.pi)*(x/np.pi))*(x/np.pi>0.5))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
        RFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
        RFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')

    Lev = args.Lev  # level of transform
    s = args.s  # dilation scale
    n = args.n  # n - 1 = Degree of Chebyshev Polynomial Approximation
    r = len(DFilters)

    # get matrix operators
    if args.Chebyshev:
        if (FrameType == 'Entropy' or FrameType == 'Sigmoid'):
            J = np.log(lambda_max / np.pi) / np.log(s)
            d = get_operator2(L, DFilters, n, s, J, Lev)
        else:
            J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1
            d = get_operator(L, DFilters, n, s, J, Lev)
        # enhance sparseness of the matrix operators (optional)
        # d[np.abs(d) < 0.001] = 0.0
        d_list = list()
        for i in range(r):
            for l in range(Lev):
                d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
    else:
        d_list = get_operator1(L, DFilters, lambdas, eigenvecs, s, Lev)
        d_list = [torch.tensor(x).to(device) for x in d_list]

    '''
    Training Scheme
    '''
    # Hyper-parameter Settings
    learning_rate = args.lr
    weight_decay = args.wd
    nhid = args.nhid

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    SaveResultFilename = 'ResultExp{0:03d}'.format(args.ExpNum)

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        record_test_acc = 0.0

        # initialize the model: setting cutoff to True makes the first layer as hard high-frequency cut-off
        if args.activation:
            model = Net(dataset.num_node_features, nhid, dataset.num_classes, r, Lev, num_nodes,
                    shrinkage=args.shrinkage, sigma=args.sigma, dropout_prob=args.dropout, activation= F.elu, cutoff1=args.cutoff1, cutoff2=args.cutoff2).to(device)
        else:
            model = Net(dataset.num_node_features, nhid, dataset.num_classes, r, Lev, num_nodes,
                        shrinkage=args.shrinkage, sigma=args.sigma, dropout_prob=args.dropout, cutoff1=args.cutoff1,
                        cutoff2=args.cutoff2).to(device)
        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # initialize the learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, d_list)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            model.eval()
            out = model(data, d_list)
            for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = out[mask].max(dim=1)[1]
                correct = float(pred.eq(data.y[mask]).sum().item())
                e_acc = correct / mask.sum().item()
                epoch_acc[i][rep, epoch] = e_acc

                e_loss = F.nll_loss(out[mask], data.y[mask])
                epoch_loss[i][rep, epoch] = e_loss

            # scheduler.step(epoch_loss['val_mask'][rep, epoch])

            # print out results
            if (epoch + 1) % 50 == 0:
                print('Epoch: {:3d}'.format(epoch + 1),
                   'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                   'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                   'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                   'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                   'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                   'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch > 10:   # and epoch < 171:
               if epoch_acc['val_mask'][rep, epoch] > max_acc:
                   #torch.save(model.state_dict(), SaveResultFilename + '.pth')
                   print('Epoch: {:3d}'.format(epoch + 1),
                         'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                         'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                         'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                         'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                         'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                         'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))
                   print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                   max_acc = epoch_acc['val_mask'][rep, epoch]
                   record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    """
    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    print('dataset:', args.dataset, '; epochs:', args.epochs, '; reps:', args.reps, '; learning_rate:', args.lr, '; weight_decay:', args.wd, '; nhid:', args.nhid,
          '; Lev:', args.Lev)
    print('s:', args.s, '; n:', args.n, '; FrameType:', args.FrameType, '; dropout:', args.dropout, '; seed:', args.seed, '; filename:', args.filename)
    print('shrinkage:', args.shrinkage, '; sigma:', args.sigma)
    print('\n')
    print(SaveResultFilename + '.pth', 'contains the saved model and ', SaveResultFilename + '.npz', 'contains all the values of loss and accuracy.')
    print('***************************************************************************************************************************')
    """
    # save the results

    message = 'Experiment {0:03d} with seed {1:5d}: Average test accuracy over {2:2d} reps: {3:.4f} with stdev {4:.4f}\n'.format(args.ExpNum,
                                                                        args.seed, num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc))
    message += 'dataset: {0}; epochs: {1:3d}; reps: {2:2d}; learning_rate: {3:.5f}; weight_decay: {4:.4f}; nhid: {5:3d}; dropout: {6:.2f};\n'.format(args.dataset,
                                                    args.epochs, args.reps, args.lr, args.wd, args.nhid, args.dropout)
    message += 'FrameType: {0}; Lev: {1:2d}; s: {2:.3f}; n: {3:1d}; Chebyshev: {4}; shrinkage: {5}; activation: {6};\n'.format(args.FrameType,
                                                    args.Lev, args.s, args.n, args.Chebyshev, args.shrinkage, args.activation)
    message += 'Frequency number: {0:3d}; sigma: {1:.3f}; alpha: {2:2f}; noiseLev: {3:.2f}; cutoff1: {4}; cutoff2: {5}\n'.format(args.FrequencyNum,
                                                   args.sigma, args.alpha, args.noiseLev, args.cutoff1, args.cutoff2)
    print('***************************************************************************************************************************')
    print(message)
    print('***************************************************************************************************************************')
    with open(SaveResultFilename + '.txt', 'w') as f:
        f.write(message)

    np.savez(SaveResultFilename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)

   
