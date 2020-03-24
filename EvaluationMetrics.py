#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:44:06 2020

@author: louisrobinson
"""

import numpy as np
from tqdm import tqdm

def RMSE(y_hat, y):#5
    return np.sqrt(np.sum(np.power(y_hat-y, 2))/len(y_hat))
    
def MAE(y_hat, y):#1
    return np.sum(np.abs(y_hat-y))/len(y_hat)

def confusion_matrix(y_hat, y, th=0.5):
    tp, fp, fn, tn = 0, 0, 0, 0
    a = np.zeros(len(y_hat))
    a[y_hat >= th] = 1
    for yh, y in zip(a, y):
        if yh==1:
            if y==1:
                tp += 1
            else:
                fp += 1
        else:
            if y==1:
                fn += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def recall_precision(y_hat, y):#2,3
    tp, fp, fn, tn = confusion_matrix(y_hat, y)
    return tp/(tp+fn), tp/(tp+fp)

def recall_precision_curve(y_hat, y, n=15):#2,3
    fpr_tpr = []
    for th in np.linspace(0,1,n):#6
        tp, fp, fn, tn = confusion_matrix(y_hat, y, th)
        fpr_tpr.append([tp/(tp+fn), tp/(tp+fp)])
    return np.array(fpr_tpr)

def ROC(y_hat, y, n=15):
    fpr_tpr = []
    for th in np.linspace(0,1,n):#6
        tp, fp, fn, tn = confusion_matrix(y_hat, y, th)
        fpr_tpr.append([fp/(fp+tn), tp/(tp+fn)])
    return np.array(fpr_tpr)

def novelty(item_pop_R):
    ''' item_list = proportion of users that rated an item '''
    # for each user: give a recommendation, produce one long list of items.
    # compute log_2(popularity(item)) / len(list)
    return -sum(np.log2(item_pop_R))/len(item_pop_R)

def catalogCoverage(rs, user_likes, user, N=1500):
    m = rs.m
    # select a context the user has been found in...
    c1s = list(user_likes[str(user)].keys())
    c1 = c1s[np.random.randint(len(c1s))]
    c2s = list(user_likes[str(user)][c1].keys())
    c2 = c2s[np.random.randint(len(c2s))]
    context = [int(c1), int(c2)]
    
    obs_items = set([])
    coverage = np.zeros(N)
    for i in tqdm(range(N)):
        recs = rs.recommend(user, list(obs_items), context, None, newUser=False)
        # since breakdown is set to None: we always recommend 10
        recs = set([r[0] for r in recs])
        obs_items = obs_items.union(recs)
        coverage[i] = len(obs_items)/m
        print(coverage[i])
    return coverage
    