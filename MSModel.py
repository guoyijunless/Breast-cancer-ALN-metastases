# from select import select
# from turtle import forward
# import numpy as np

import warnings

warnings.filterwarnings("ignore")

# import datetime, time
# import random
# import copy

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score, recall_score, f1_score 

import torch
from torch import nn
# from torch.nn import functional as F

class DynamicConv2D(nn.Module):
        def __init__(self, in_plane, out_plane=32, k=8) -> None:
            super().__init__()
            self.k = k
            self.conv = nn.Sequential(nn.Conv2d(in_plane, out_plane, [3, 3], padding=1),
            nn.BatchNorm2d(out_plane),
            nn.LeakyReLU(inplace=True))
            self.avgp1= nn.AdaptiveAvgPool2d((1, 1))
            self.dense1 = nn.Sequential(nn.Linear(in_plane, 2 * k), 
            nn.LeakyReLU(inplace=True))
            self.dense2 = nn.Linear(2 * k, k)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            att = self.avgp1(x)
            att = torch.flatten(att, 1)
            att = self.dense1(att)
            att = self.dense2(att)

            att = self.softmax(att/0.7)

            feature = att[:, 0] * torch.permute(self.conv(x), [1, 2, 3, 0])
            for i in range(1, self.k):
                feature += att[:, i] * torch.permute(self.conv(x), [1, 2, 3, 0])
            
            feature = torch.permute(feature, [3, 0, 1, 2])
        
            return feature



class Scale_Block(nn.Module):
    def __init__(self, input_chnl, filters, input_plane, out_plane, pool_size=None, pool_strides=None, enter=False, k=8) -> None:
        super().__init__()
        self.enter = enter
        self.maxpool1 = nn.MaxPool2d(pool_size, pool_strides)
        self.conv = nn.Sequential(nn.Conv2d(input_chnl, filters, [3, 3], padding=1),
                                    nn.BatchNorm2d(filters),
                                    nn.LeakyReLU(inplace=True))
        if not self.enter:                            
            self.maxpool2 = nn.MaxPool2d([2, 2], [2, 2])
        self.dynamicconv = DynamicConv2D(input_plane, out_plane=out_plane, k=k)
    
    def forward(self, x, last_conv=None):
        if not self.enter:
            x = self.maxpool1(x)

        conv = self.conv(x)

        if not self.enter:
            q = torch.concat([conv, self.maxpool2(last_conv)], dim=1)
            conv = self.dynamicconv(q)
        else:
           conv = self.dynamicconv(conv) 

        return conv

class MSModel(nn.Module):
    def __init__(self, in_planes, scale_range=5, dropout=None, asextractor=True) -> None:
        super().__init__()
        k = 16
        k_dyn = 8
        self.asextractor = asextractor
        # self.enter = nn.BatchNorm2d(in_planes)
        self.scale_range = scale_range
        # self.scale1 = Scale_Block(k, k, enter=True)
        # self.scale2 = Scale_Block(k, k, [2, 2], [2, 2])
        # self.scale3 = Scale_Block(k, k, [4, 4], [4, 4])
        # self.scale4 = Scale_Block(k, k, [8, 8], [8, 8])
        # self.scale5 = Scale_Block(k, k, [16, 16], [16, 16])
        self.scale_blocks = []
        for i in range(self.scale_range):
            if i == 0:
                self.scale_blocks.append(Scale_Block(in_planes, k, k, k, enter=True, k=k_dyn))
            else:
                pool_size = [2**i, 2**i]
                self.scale_blocks.append(Scale_Block(in_planes, k, 2*k, 2*k, pool_size, pool_size, k=k_dyn))
                k = 2*k
        self.scale_blocks = nn.ModuleList(self.scale_blocks)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = dropout
        if self.dropout is not None:
            self.dropout = nn.Dropout()
        self.classifier = nn.Linear(496, 2)

       
    
    def forward(self, x):
        # x = self.enter(x)
        # conv1 = self.scale1(x)
        # conv2 = self.scale2(x)
        # conv3 = self.scale3(x)
        # conv4 = self.scale4(x)
        # conv5 = self.scale5(x)
        poolings = []
        last_conv = None
        for scale_block in self.scale_blocks:
            conv = scale_block(x, last_conv=last_conv)
            last_conv = conv
            # print(conv.shape)
            poolings.append(torch.flatten(self.avgpool(conv), 1))
        
        feature = torch.concat(poolings, dim=-1)
        if self.dropout is not None:
            feature = self.dropout(feature)
        if not self.asextractor:
            feature = self.classifier(feature)

        return feature
    

if __name__ == '__main__':
    from torchvision.models import resnet18
    net = MSModel(3)
    print(net)
    x = torch.rand(1, 3, 224, 224)
    print(net(x).shape)

        

            



    
    
    

    
        