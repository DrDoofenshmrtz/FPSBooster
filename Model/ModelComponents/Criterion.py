# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:27:35 2022

@author: ABHRANIL
"""

import torch

def Criterion(y, y_pred):         
    # y and y_pred should be inputs of shape (n_C, H, W)
    num_channels = y.shape[0]
    loss = 0
    for i in range(num_channels):
        loss += torch.linalg.matrix_norm(y[i] - y_pred[i])
    loss = loss / num_channels
    return loss