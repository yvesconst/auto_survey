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


# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# def next_batch(inputs, targets, batchSize):
#  	# loop over the dataset
#  	for i in range(0, inputs.shape[0], batchSize):
#          yield (inputs[i:i + batchSize], targets[i:i + batchSize])

# df_student = pd.read_csv("./dataset/merged_student_engagement_level.csv")

# df_test =  df_student.copy()


# df_test['engagement'] = df_test['engagement'].map(
#     {'H': 2, 'M': 1, 'L': 0}
# )

# X = df_test.loc[:,["reviews", "a1", "a2", "a3", "gender", "grade"]].values
# y = df_test.iloc[:, 4].values


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# X_train = X_train[0:700]
# y_train = y_train[0:700]

# X_test = T.FloatTensor(X_test)
# y_train = T.LongTensor(y_train)
# y_test = T.LongTensor(y_test)

# scaler = StandardScaler()

# X_train = T.from_numpy(scaler.fit_transform(X_train)).float()


# epochs = 50
# BATCH_SIZE = 10
# losses = []
# model = ClassificationNet().to(device)
# optimizer = T.optim.Adam(model.parameters(), lr=0.001)
# criterion = T.nn.CrossEntropyLoss()
  

# j = 0
# for (batchX, batchY) in next_batch(X_train, y_train, BATCH_SIZE):
#     (batchX, batchY) = (batchX, batchY.to(device))
#     j += 1
    
#     for i in range(epochs):
#         i += 1
        
#         y_pred = model(batchX)
#         loss = criterion(y_pred, batchY.long())
        
#         losses.append(loss)
        
#         if j % 5 == 0:
#             print(f'epoch: {i} -> batch: {j} -> loss: {loss}')
            
#         optimizer.zero_grad()
#         loss.backward()
#         T.nn.utils.clip_grad_norm_(model.parameters(), 5)
#         optimizer.step()
        
# # X_test = X_test[0:500]
# # y_test = y_test[0:500]

# X_test = T.from_numpy(scaler.fit_transform(X_test)).float()

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

 
# with T.no_grad():
#     model.eval()
#     pred = model(X_test)
#     _, pr = T.max(pred, 1)
    
#     print(pr[0:50])
#     print(y_test[0:50])
    
#     conf_mat = confusion_matrix(pr.tolist(), y_test.tolist())
#     sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    #confusion_matrix = zip(pr, y_test.item())
    #print(confusion_matrix)

#nb_classes = len(y_test)
# confusion_matrix = T.zeros(nb_classes, nb_classes)
# with T.no_grad():
#     model.eval()
#     for k, yy in enumerate(y_test):
#         inputs = X_test[k].to(device)
#         classes = yy.to(device)
#         outputs = model(inputs)
#         _, preds = T.max(outputs, 1)
#         for t, p in zip(classes.view(-1), preds.view(-1)):
#                 confusion_matrix[t.long(), p.long()] += 1

#print(confusion_matrix)