import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from utils import preprocess_adj_new, preprocess_adj_new1


class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_xdims, n_hid, n_out, adj_A, mask, num_nodes, do_prob=0., tol = 0.1):
        super(MLPEncoder, self).__init__()

        self.n_hid = n_hid
        self.dropout_prob = do_prob
        self.num_nodes = num_nodes
        self.mask = mask

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        
        # Either BatchNorm or LayerNorm
        self.bn1 = nn.BatchNorm1d(n_hid)
        # self.ln1 = nn.LayerNorm(n_hid)

        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # Apply mask
        adj_A1 *= self.mask

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)
        H1 = self.fc1(inputs)

        # BatchNorm
        H1 = H1.view(-1, self.n_hid)
        H1 = self.bn1(H1)
        H1 = H1.view(-1, self.num_nodes, self.n_hid)

        # LayerNorm
        # H1 = self.ln1(H1)

        H1 = F.relu(H1)
        H1 = F.dropout(H1, p=self.dropout_prob, training=self.training)
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        return logits, adj_A1, self.z, self.z_positive, self.adj_A, self.Wa


class SEMEncoder(nn.Module):
    """SEM encoder module."""
    def __init__(self, n_in, adj_A):
        super(SEMEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad = True))

        self.Wa = torch.zeros(n_in, dtype=torch.double)

    def init_weights(self):
        nn.init.xavier_normal(self.adj_A.data)

    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_A = I-A^T, adj_A_inv = (I-A^T)^(-1)
        adj_A = preprocess_adj_new((adj_A1))
        adj_A_inv = preprocess_adj_new1((adj_A1))

        meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_A, inputs), 0))
        logits = torch.matmul(adj_A, inputs-meanF)

        return logits, adj_A1, None, None, self.adj_A, self.Wa


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        H3 = F.dropout(H3, p=self.dropout_prob, training=self.training)
        out = self.out_fc2(H3)

        return out

class SEMDecoder(nn.Module):
    """SEM decoder module."""

    def __init__(self):
        super(SEMDecoder, self).__init__()
        print('Using learned interaction net decoder.')

    def forward(self, input_z, origin_A, Wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa)

        return mat_z-Wa

