#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:44:06 2020

@author: louisrobinson
"""

import numpy as np


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

def ROC(y_hat, y, n=15):
    fpr_tpr = []
    for th in np.linspace(0,1,n):#6
        tp, fp, fn, tn = confusion_matrix(y_hat, y, th)
        fpr_tpr.append([fp/(fp+tn), tp/(tp+fn)])
    return np.array(fpr_tpr)

def novelty(p_item_list, popularity):
    ''' item_list = proportion of users that rated an item '''
    # for each user: give a recommendation, produce one long list of items.
    # compute log_2(popularity(item)) / len(list)
    return -sum(np.log2(p_item_list))/len(p_item_list)

def item_coverage(all_items, predictable_items):#predictable_items must have say > 10 ratings
    return len(predictable_items)/len(all_items)

'''Catalog coverage is the percentage of recommended user-item pairs over the 
total number of potential pairs. The number of recommended user-item pairs can 
be represented by the length of the recommender lists L.'''
def catelog_coverage(list_length, num_ratings):
    return list_length/num_ratings
    