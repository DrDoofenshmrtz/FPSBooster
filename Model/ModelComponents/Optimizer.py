# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 21:05:15 2022

@author: ABHRANIL
"""

import torch

def Optimizer(model, learning_rate):
  return torch.optim.Adam(model.parameters(), lr = learning_rate)