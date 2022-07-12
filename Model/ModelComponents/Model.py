# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:10:36 2022

@author: ABHRANIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layer_1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 1)
        self.conv_layer_2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 1)
        self.conv_layer_3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1)
        
        self.act_layer = nn.ReLU()
        
        self.pool_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.up_conv_layer_1 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.up_conv_layer_2 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2)                    
                                              
    def forward(self, x):
        z1 = self.conv_layer_1(x)
        a1 = self.act_layer(z1)
        a1 = self.pool_layer(a1)
        z2 = self.conv_layer_2(a1)
        a2 = self.act_layer(z2)
        y1 = a2                                # saving activation before pooling for feature extraction
        a2 = self.pool_layer(a2)
        z3 = self.conv_layer_3(a2)
        a3 = self.act_layer(z3)
        y2 = a3                                # saving activation for feature extraction
        z4 = self.up_conv_layer_1(a3)
        a4 = self.act_layer(z4)
        z5 = self.up_conv_layer_2(a4)
        a5 = self.act_layer(z5)
        y = a5
        
        return y, y1, y2
    
class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.auto_encoder = AutoEncoder()
        
        # layer for upscaling second activation obtained from output of encoder
        self.up_scale_layer_1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 2, stride = 2)
    
        # layer for upscaling third activation obtained from output of encoder
        self.up_scale_layer_2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 32, kernel_size = 4, stride = 4)
    
        # main convolutional model
        self.conv_layer = nn.Conv2d(in_channels = 192, out_channels = 3, kernel_size = 1)
        self.act_layer = nn.ReLU()
    
        def forward(self, x1, x2):
            
            y1, y1_1, y1_2 = self.auto_encoder(x1)
            y2, y2_1, y2_2 = self.auto_encoder(x1)
            
            y = torch.concat((y1, y2), dim = 0)
            
            y1_1 = self.up_scale_layer_1(y1_1)
            y2_1 = self.up_scale_layer_1(y2_1)
            y_1 = torch.concat((y1_1, y2_1), dim = 0)
            
            y1_2 = self.up_scale_layer_1(y1_2)
            y2_2 = self.up_scale_layer_1(y2_2)
            y_2 = torch.concat((y1_2, y2_2), dim = 0)
            
            x = torch.concat((y, y_1, y_2), dim = 0)
            
            z1 = self.conv_layer(x)
            a1 = self.act_layer(z1)
            out = a1
            
            return out
        
def obtain_model(device = torch.device('cpu')):
    return Model().to(device)