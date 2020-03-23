#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:22:15 2020

@author: louisrobinson
"""
from Technique1_SVD import SVD
from Technique2_TF import TF
import matplotlib.pyplot as plt
from tqdm import tqdm
from EvaluationMetrics import RMSE, MAE, recall_precision, ROC, novelty, item_coverage, catelog_coverage
import numpy as np


rs = SVD(train=True)
rs.load()

''' TEST '''
difference, num = [], 1000
for _ in range(num):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    [i, j] = indices[:2]
    F_hat = 1 if rs.F(i,j)>0.5 else 0
    difference.append(np.abs(F_hat-rs.y(i,j)))
print(1-sum(difference)/num)

num = 100000
ys, fs = np.zeros(num), np.zeros(num)
for s in tqdm(range(num)):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    [i, j] = indices[:2]
    fs[s], ys[s] = rs.F(i,j), rs.y(i,j)
    
frounded = np.round(fs)

rmse, mae = RMSE(frounded, ys), MAE(frounded, ys)
recall, precision = recall_precision(frounded, ys)
roc_svd = ROC(fs, ys, 20)
#novelty(p_item_list)
#item_coverage(all_items, predictable_items)

print('(rmse, mae, recall, precision)')
print((rmse, mae, recall, precision))




rs = TF(train=True)
rs.load()



''' TEST '''
difference, num = [], 1000
for _ in range(num):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    [i, j], c_idxs = indices[:2], list(indices[2:])
    F_hat = 1 if rs.F(i,j,c_idxs)>0.5 else 0
    difference.append(np.abs(F_hat-rs.y(i,j,c_idxs)))
print(1-sum(difference)/num)



num = 100000
ys, fs = np.zeros(num), np.zeros(num)
for s in tqdm(range(num)):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs.Y_coords[choice][rs.test_idxs[choice][np.random.randint(rs.n_te[choice])]]
    [i, j], c_idxs = indices[:2], list(indices[2:])
    fs[s], ys[s] = rs.F(i,j,c_idxs), rs.y(i,j,c_idxs)
    
frounded = np.round(fs)

rmse, mae = RMSE(frounded, ys), MAE(frounded, ys)
recall, precision = recall_precision(frounded, ys)
roc_tf = ROC(fs, ys, 20)
#novelty(p_item_list)
#item_coverage(all_items, predictable_items)

print('(rmse, mae, recall, precision)')
print((rmse, mae, recall, precision))




plt.clf()
plt.plot(roc_svd[:,0], roc_svd[:,1], 'r', label='SVD')
plt.plot(roc_tf[:,0], roc_tf[:,1], "#72246C", label='TF')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('ROC', dpi=400)
plt.show()