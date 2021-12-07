import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        #temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        temp = self.conv1(X) + self.conv2(X)
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class GCNBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(GCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))#nn.Parameter会自动被认为是module的可训练参数
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        #return t3

class GCNBlock2(nn.Module):
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(GCNBlock2, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels, spatial_channels))#nn.Parameter会自动被认为是module的可训练参数
        #self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        #t3 = self.temporal2(t2)
        return self.batch_norm(t2)
        #return t2


class GCNS(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,num_timesteps_output):
        super(GCNS, self).__init__()
        self.block1 = GCNBlock(in_channels=num_features, out_channels=64,
                               spatial_channels=16, num_nodes=num_nodes)
        self.block2 = GCNBlock(in_channels=64, out_channels=64,
                               spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 0 * 2) * 64, num_timesteps_output)


    def forward(self, A_hat, X):
        #print("x_size: ",X.shape)
        out1 = self.block1(X, A_hat)
        #print("out1_size: ",out1.shape)
        #out2 = self.block2(out1, A_hat)
        #print("out2_size: ", out2.shape)
        out3 = self.last_temporal(out1)
        #print("out3_size: ", out3.shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))) # (a,b,c,d)--reshape(a,b,-1)-->(a,b,abcd/ab)
        #print("out4_size: ", out4.shape)
        return out4

class GCNS2(nn.Module):
    def __init__(self, num_nodes, num_features, num_timesteps_input,num_timesteps_output):
        super(GCNS2, self).__init__()
        self.block1 = GCNBlock2(in_channels=num_features, out_channels=64,
                                spatial_channels=16, num_nodes=num_nodes)
        '''
        timeblock中，kernel_size=2引导的维度降低1
        block1 包含2次timeblock block2 包含2次timeblock last_temporal 包含1次timeblock
        所以，在Linear层之前经历了5次由kernel_size=3引导的维度降低
        '''
        self.fully = nn.Linear(num_timesteps_input * 16, num_timesteps_output)

    def forward(self, A_hat, X):
        #print("x_size: ",X.shape)
        out1 = self.block1(X, A_hat)
        #print("out1_size: ",out1.shape)
        #out2 = self.block2(out1, A_hat)
        #print("out2_size: ", out2.shape)
        #out3 = self.last_temporal(out2)
        #print("out3_size: ", out3.shape)
        out4 = self.fully(out1.reshape((out1.shape[0], out1.shape[1], -1))) # (a,b,c,d)--reshape(a,b,-1)-->(a,b,abcd/ab)
        #print("out4_size: ", out4.shape)
        return out4
