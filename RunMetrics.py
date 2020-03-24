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
from EvaluationMetrics import RMSE, MAE, recall_precision, ROC, novelty, catalogCoverage, recall_precision_curve
import numpy as np


###############################################################################
###################    Singular Value Decomposition RS    #####################
###############################################################################

rs1 = SVD(train=True)
rs1.load()

''' TEST '''
difference, num = [], 1000
for _ in range(num):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs1.Y_coords[choice][rs1.test_idxs[choice][np.random.randint(rs1.n_te[choice])]]
    [i, j] = indices[:2]
    F_hat = 1 if rs1.F(i,j)>0.5 else 0
    difference.append(np.abs(F_hat-rs1.y(i,j)))
print(1-sum(difference)/num)

num = 100000
ys, fs = np.zeros(num), np.zeros(num)
for s in tqdm(range(num)):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs1.Y_coords[choice][rs1.test_idxs[choice][np.random.randint(rs1.n_te[choice])]]
    [i, j] = indices[:2]
    fs[s], ys[s] = rs1.F(i,j), rs1.y(i,j)
    
frounded = np.round(fs)
rmse, mae = RMSE(frounded, ys), MAE(frounded, ys)
#rmse, mae = RMSE(fs, ys), MAE(fs, ys)
recall, precision = recall_precision(frounded, ys)
roc_svd = ROC(fs, ys, 20)
pr_svd = recall_precision_curve(fs, ys, 20)
#novelty(p_item_list)
#item_coverage(all_items, predictable_items)

print('(rmse, mae, recall, precision)')
print((rmse, mae, recall, precision))



###############################################################################
###################           Catalog Coverage            #####################
###############################################################################
user, N = 0, 1500
coverage_svd = catalogCoverage(rs1, rs1.user_likes, user, N=N)

#plt.plot([i*10 for i in range(1, N+1)], coverage_svd, label='Catelog coverage')
#plt.plot([rs.m, rs.m], [0, 1.05], 'r', label='Total number of items')
#plt.xlabel('Number of tracks recommended')
#plt.ylabel('Proportion of tracks seen')
#plt.ylim([0, 1.05])
#plt.legend()
#plt.savefig('CatalogCoverage', dpi=300)
#plt.show()


###############################################################################
###################                Novelty                #####################
###############################################################################
N = 500
user = 0
m = rs1.m
# select a context the user has been found in...
c1s = list(rs1.user_likes[str(user)].keys())
c1 = c1s[np.random.randint(len(c1s))]
c2s = list(rs1.user_likes[str(user)][c1].keys())
c2 = c2s[np.random.randint(len(c2s))]
context = [int(c1), int(c2)]

obs_items = set([])
nov1 = np.zeros(N)
for i in tqdm(range(N)):
    recs = rs1.recommend(user, list(obs_items), context, None, newUser=False)
    # since breakdown is set to None: we always recommend 10
    recs = list(set([r[0] for r in recs]))
    item_pop_R = rs1.item_popularity[recs]
    nov1[i] = novelty(item_pop_R)
#    print(nov1[i])
print(np.mean(nov1))
print(np.std(nov1))





###############################################################################
###################       Tensor Factorisation RS         #####################
###############################################################################

rs2 = TF(train=True)
rs2.load()



''' TEST '''
difference, num = [], 1000
for _ in range(num):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs2.Y_coords[choice][rs2.test_idxs[choice][np.random.randint(rs2.n_te[choice])]]
    [i, j], c_idxs = indices[:2], list(indices[2:])
    F_hat = 1 if rs2.F(i,j,c_idxs)>0.5 else 0
    difference.append(np.abs(F_hat-rs2.y(i,j,c_idxs)))
print(1-sum(difference)/num)



num = 100000
ys, fs = np.zeros(num), np.zeros(num)
for s in tqdm(range(num)):
    choice = 1 if np.random.random() < 0.2 else 0 # do positive example:
    indices = rs2.Y_coords[choice][rs2.test_idxs[choice][np.random.randint(rs2.n_te[choice])]]
    [i, j], c_idxs = indices[:2], list(indices[2:])
    fs[s], ys[s] = rs2.F(i,j,c_idxs), rs2.y(i,j,c_idxs)
    
frounded = np.round(fs)

rmse, mae = RMSE(frounded, ys), MAE(frounded, ys)
recall, precision = recall_precision(frounded, ys)
roc_tf = ROC(fs, ys, 20)
pr_tf = recall_precision_curve(fs, ys, 20)
#novelty(p_item_list)
#item_coverage(all_items, predictable_items)

print('(rmse, mae, recall, precision)')
print((rmse, mae, recall, precision))


###############################################################################
###################            Both ROC curves            #####################
###############################################################################

plt.clf()
plt.plot(roc_svd[:,0], roc_svd[:,1], 'c', label='SVD-PF')
plt.plot(roc_tf[:,0], roc_tf[:,1], "#72246C", label='TF-DPP')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('ROC', dpi=400)
plt.show()

###############################################################################
###################     Both Precision-Recall curves      #####################
###############################################################################

plt.clf()
plt.plot(pr_svd[:,0], pr_svd[:,1], 'c', label='SVD-PF')
plt.plot(pr_tf[:,0], pr_tf[:,1], "#72246C", label='TF-DPP')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('PR', dpi=400)
plt.show()

###############################################################################
###################           Catalog Coverage            #####################
###############################################################################
user, N = 0, 1500
coverage_tf = catalogCoverage(rs2, rs1.user_likes, user, N=N)

plt.clf()
plt.plot([i*10 for i in range(1, N+1)], coverage_svd, 'c', label='SVD-PF')
plt.plot([i*10 for i in range(1, N+1)], coverage_tf, "#72246C", label='TF-DPP')
plt.plot([rs2.m, rs2.m], [0, 1.05], 'r', label='Total number of items')
plt.xlabel('Number of tracks recommended')
plt.ylabel('Proportion of tracks seen')
plt.ylim([0, 1.05])
plt.legend()
#plt.title('Catelog coverage')
plt.savefig('CatalogCoverage', dpi=300)
plt.show()


###############################################################################
###################                Novelty                #####################
###############################################################################
N = 500
user = 0
m = rs2.m
# select a context the user has been found in...
c1s = list(rs1.user_likes[str(user)].keys())
c1 = c1s[np.random.randint(len(c1s))]
c2s = list(rs1.user_likes[str(user)][c1].keys())
c2 = c2s[np.random.randint(len(c2s))]
context = [int(c1), int(c2)]

obs_items = set([])
nov2 = np.zeros(N)
for i in tqdm(range(N)):
    recs = rs2.recommend(user, list(obs_items), context, None, newUser=False)
    # since breakdown is set to None: we always recommend 10
    recs = list(set([r[0] for r in recs]))
    item_pop_R = rs2.item_popularity[recs]
    nov2[i] = novelty(item_pop_R)
#    print(nov[i])
print(np.mean(nov2))
print(np.std(nov2))
    
    