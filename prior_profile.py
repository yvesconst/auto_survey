#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 05:35:37 2022

@author: ytadjota
"""

import torch as T
import pandas as pd
import torch.nn as nn


device = "cuda" if T.cuda.is_available() else "cpu"

class ClassificationNet(T.nn.Module):
  def __init__(self, in_features=6, out_features=3, n_per_layer=10):
    super(ClassificationNet, self).__init__()
    self.fc1 = nn.Linear(in_features, n_per_layer) 
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(n_per_layer, n_per_layer) 
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(n_per_layer, out_features) 

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu1(out)
    out = self.fc2(out)
    out = self.relu2(out)
    out = self.fc3(out)
    return out
