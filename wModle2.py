import numpy as np
import scipy as sp
import torch
from captum.attr import IntegratedGradients
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv ,GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import os
import math
from collections import OrderedDict

import numpy as np
import torchvision.models as models
from GAT import GAT
from GraphAttentionLayer import GraphAttentionLayer
from sklearn.manifold import TSNE

class wModel2(torch.nn.Module):
    def __init__(self):
        super(wModel2, self).__init__()
        self.GCNconv1 = GCNConv(81, 10)
        self.GraphA = GAT(30,30,30,0.2,0.2,4)
        dims = 7 * 3  # 所有头总共需要的输入维度
        heads = 3  # 单注意力头的总共个数
        dropout_pro = 0.0  # 单注意力头
        self.attentionLayer = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=heads)
        self.linear_kmer = torch.nn.Linear(84, 30)
        self.linear_vec = torch.nn.Linear(72, 81)
        # self.linear_DPCP_vec = torch.nn.Linear(30, 10)
        self.linear_out = torch.nn.Linear(1000, 1)
        self.x01_linear1 = torch.nn.Linear(21, 21)
        self.linear_x_DPCP =torch.nn.Linear(7, 5)
        self.linear1 = torch.nn.Linear(21, 1)
        self.linear2 = torch.nn.Linear(81, 2)
        self.convx_gat = nn.Conv1d(81, 30, kernel_size=21)
        self.convx_out = nn.Conv1d(81, 4, kernel_size=60)
        self.BN =nn.BatchNorm1d(10)
        self.BN30 = nn.BatchNorm1d(30)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.RES = models.resnet18(pretrained=True)
        self.RES.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#RES能处理单通道的序列特征
        self.maxpool2d_t_p = nn.MaxPool2d(kernel_size=(1,30), stride=1, padding=0, dilation=1)
        self.maxpool2d_t_end = nn.MaxPool2d(kernel_size=(4,60), stride=1, padding=0, dilation=1)
        self.tsne = TSNE(n_components=2)
        self.XCNN = nn.Conv1d(81, 81, kernel_size=37)
        self.linear_out1 = torch.nn.Linear(60, 30)
        self.linear_out2 = torch.nn.Linear(30, 1)
    def forward(self,data,epoch,TrainOrPre):
        #data:
        # 0：101,84-kmer
        # 1: 92,30 -vec
        # 2: 101,101-pair
        x_Kmer = data["Kmer"].to(torch.float32)#81.84
        x_DPCP = data["DPCP"].to(torch.float32)#81.6
        x_pair = data["pair"].to(torch.float32)#81.81
        x_vec = data["vec"].to(torch.float32)#72.30
        x_Kmer = self.linear_kmer(x_Kmer)#81.30
        x_vec = x_vec.permute(0,2,1)
        x_vec = self.linear_vec(x_vec)
        x_vec = x_vec.permute(0, 2, 1)#81.30
        x_gat = torch.zeros_like(x_Kmer)
        for i in range(x_pair.size()[0]):
            a = x_pair[i]
            # a[a > 0.8] = 1
            # a[a <= 0.8] = 0
            e = x_Kmer[i]
            x_gat_1 = self.GraphA(e, a)
            x_gat_1 = self.BN30(x_gat_1)
            x_gat_1 = self.relu(x_gat_1)
            x_gat[i]=x_gat_1
        x_out = torch.cat((x_Kmer, x_DPCP,x_vec), dim=2)#81.66
        x_cnn = self.XCNN(x_out)
        x_out = torch.cat((x_gat,x_cnn), dim=2)#81.30+81.30
        #%%做散点图
        # if epoch == 39:
        #     posList = []
        #     negList = []
        #     for i in range(Y.size(0)):
        #         if Y[i] == 1:
        #             posList.append(i)
        #         else:
        #             negList.append(i)
        #     posList_data = []
        #     negList_data = []
        #     for i in range(x_Kmer.size(0)):
        #         if i in posList:
        #             if len(posList_data) == 0:
        #                 posList_data = x_Kmer[i].unsqueeze(0)
        #             else:
        #                 posList_data = torch.cat((posList_data, x_Kmer[i].unsqueeze(0)), 0)
        #         else:
        #             if len(negList_data) == 0:
        #                 negList_data = x_Kmer[i].unsqueeze(0)
        #             else:
        #                 negList_data = torch.cat((negList_data, x_Kmer[i].unsqueeze(0)), 0)
        #     posList_data = self.maxpool2d_t_p(posList_data)
        #     posList_data = posList_data.squeeze()
        #     posList_data = self.tsne.fit_transform(posList_data.cpu().detach().numpy())
        #     negList_data = self.maxpool2d_t_p(negList_data)
        #     negList_data = negList_data.squeeze()
        #     negList_data = self.tsne.fit_transform(negList_data.cpu().detach().numpy())
        #     posList_data = torch.from_numpy(posList_data)
        #     negList_data = torch.from_numpy(negList_data)
        #     posListEnd_data = []
        #     negListEnd_data = []
        #     for i in range(x_out.size(0)):
        #         if i in posList:
        #             if len(posListEnd_data) == 0:
        #                 posListEnd_data = x_out[i].unsqueeze(0)
        #             else:
        #                 posListEnd_data = torch.cat((posListEnd_data, x_out[i].unsqueeze(0)), 0)
        #         else:
        #             if len(negListEnd_data) == 0:
        #                 negListEnd_data = x_out[i].unsqueeze(0)
        #             else:
        #                 negListEnd_data = torch.cat((negListEnd_data, x_out[i].unsqueeze(0)), 0)
        #     posListEnd_data = self.maxpool2d_t_end(posListEnd_data)
        #     posListEnd_data = posListEnd_data.squeeze()
        #     posListEnd_data = self.tsne.fit_transform(posListEnd_data.cpu().detach().numpy())
        #     negListEnd_data = self.maxpool2d_t_end(negListEnd_data)
        #     negListEnd_data = negListEnd_data.squeeze()
        #     negListEnd_data = self.tsne.fit_transform(negListEnd_data.cpu().detach().numpy())
        #     posListEnd_data = torch.from_numpy(posListEnd_data)
        #     negListEnd_data = torch.from_numpy(negListEnd_data)
        #     torch.set_printoptions(precision=4,sci_mode=False)
        #     protein = "AUF1"
        #     with open('./Datasets/circRNA-RBP/' + protein + '/posList_data.txt', 'a') as f:
        #         f.write('F1 ' + str(posList_data) + '\n')
        #     with open('./Datasets/circRNA-RBP/' + protein + '/negList_data.txt', 'a') as f:
        #         f.write('F1 ' + str(negList_data) + '\n')
        #     with open('./Datasets/circRNA-RBP/' + protein + '/posListEnd_data.txt', 'a') as f:
        #         f.write('F1 ' + str(posListEnd_data) + '\n')
        #     with open('./Datasets/circRNA-RBP/' + protein + '/negListEnd_data.txt', 'a') as f:
        #         f.write('F1 ' + str(negListEnd_data) + '\n')
        # %%
        x_out = self.linear_out1(x_out)
        x_out = self.linear_out2(x_out)
        x_out =x_out.sum(axis=1).squeeze()
        x_out = self.sigmoid(x_out)

        return x_out  # 759x1
