#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:18:27 2020

@author: louisrobinson
"""
import numpy as np
from DPP import  DualDPP
import json

def load_data():
    ''' [user, item, mn, hr, location, r] '''
    ''' POP_RND we select 9 tracks randomly that the user has not listened to '''
    with open("data/rdn_train_final.json", 'r') as f: rdn_train = json.load(f)
    with open("data/rdn_test_final.json", 'r') as f: rdn_test = json.load(f)
    return (np.array(rdn_train, dtype=int), np.array(rdn_test, dtype=int))
    
def conv(data):
    ''' take first two cols, last col, and rewrite col 5 (location to be discretised) '''
    new_data = np.zeros((len(data), 5), dtype=int)
    new_data[:, [0,1,4]] = data[:, [0,1,5]]
    
    mnths = {mnth:i for i, mnth in enumerate(range(1, 13))}
    locs = [-12,-11,-10,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,13]
    locs = {l:i for i, l in enumerate(locs)}
    for i in range(len(data)):
        new_data[i, 2] = locs[data[i, 4]]
        new_data[i, 3] = mnths[data[i, 2]]
    return new_data

def makeY(data, n, m):
    all_ = set(range(len(data)))
    pos = set(list(np.nonzero(data[:, -1])[0]))
    neg = list(all_ - pos)
    Y_coords = {0:data[neg, :4], 1:data[list(pos), :4]}
    item_popularity = np.zeros(m)
    Y_tele = dict()
#    user_likes = {str(i):dict() for i in range(n)}
    for [i, j, k, l, r] in data:
        item_popularity[i] += 1
        if r==1:
#            if str(k) in user_likes[str(i)]:
#                if str(l) in user_likes[str(i)][str(k)]:
#                    user_likes[str(i)][str(k)][str(l)].add(j)
#                else:
#                    user_likes[str(i)][str(k)][str(l)] = set([j])
#            else:
#                user_likes[str(i)][str(k)] = {str(l):set([j])}
            if i in Y_tele:
                if j in Y_tele[i]:
                    if k in Y_tele[i][j]:
                        Y_tele[i][j][k].add(l)
                    else:
                        Y_tele[i][j][k] = set([l])
                else:
                    Y_tele[i][j] = {k:set([l])}
            else:
                Y_tele[i] = {j:{k:set([l])}}
    # convert all sets in user_likes to lists so that it is json serialisable
#    for i, ui in user_likes.items():
#        for k, uik in ui.items():
#            for l, uikl in uik.items():
#                user_likes[str(i)][str(k)][str(l)] = [str(j) for j in uikl]
    return Y_tele, Y_coords, item_popularity, len(pos), len(neg)


class TF:
    def __init__(self, train=False):
        self.n_contexts = 2

        self.mtx = lambda a, b: np.random.normal(0, 0.05, (a, b))
        
        if train:
            print('loading data...')
            rdn_tr, rdn_te = load_data()#, usr_tr, usr_te
            # user, item, mn, hr, time_shift, r
            # time_shift is an approximation of location
            print('transforming data...')
        
            dataset1 = conv(rdn_tr)
            dataset2 = conv(rdn_te)
            dataset = np.concatenate((dataset1, dataset2), axis=0)
            
            self.n = len(set(list(dataset[:,0])))# num users, approx 2324
            self.m = len(set(list(dataset[:,1])))# num tracks, approx 7886
            self.cs = [len(set(list(dataset[:,2]))), len(set(list(dataset[:,3])))]# num locations, 24, num months, 12
            # ^ dims of tensor Y (only partially stored in memory).
            self.du, self.dm, self.dcs = 10, 15, [4, 3]# latent dimensions
        
            self.Y_tele, self.Y_coords, self.item_popularity, self.n_pos, self.n_neg = makeY(dataset, self.n, self.m)
            np.save('TF-model/item_popularity', self.item_popularity)
#            self.save_pred()
            
            self.tr_te_split()
            
            self.U, self.M, self.Cs = self.mtx(self.n, self.du), self.mtx(self.m, self.dm), [self.mtx(c, dc) for c, dc in zip(self.cs, self.dcs)]
            self.S = np.random.normal(0, 0.05, tuple([self.du, self.dm] + self.dcs))
        else:
            print('loading model...')
            self.load(n_contexts=2)
            print('done.')

    
    def rec_y(self, queue, dict_):
        if len(queue)==0: return 1
        ii = queue.pop(0)
        if ii in dict_:
            return self.rec_y(queue, dict_[ii])
        return 0
    
    def y(self, i, j, c_idxs):
        ''' func: y can retrieve all rating for 3-tensor user-item-location.
              however it does not distinguish between observed an unobserved items,
              Y_coords contains the examples '''
        return self.rec_y(c_idxs, self.Y_tele)
    
    def save(self, path='TF-model/'):
        np.save(path+'U', self.U), np.save(path+'M', self.M), np.save(path+'S', self.S)
        for i, C in enumerate(self.Cs): np.save(path+'C'+str(i), C)
        
    def load(self, n_contexts=2, path='TF-model/'):
        self.U, self.M, self.S = np.load(path+'U.npy'), np.load(path+'M.npy'), np.load(path+'S.npy')
        self.Cs = [np.load(path+'C'+str(i)+'.npy') for i in range(n_contexts)]
        (self.n, self.du), (self.m, self.dm) = self.U.shape, self.M.shape
        [self.cs, self.dcs] = list(zip(*[C.shape for C in self.Cs]))
        self.item_popularity = np.load(path+'item_popularity.npy')
    
    def rec_dot(self, queue, out):
        if len(queue)==0: return out
        (v, ax) = queue.pop(0)
        return self.rec_dot(queue, np.tensordot(out, v, axes=([ax], [0])))
    
    def F(self, i, j, c_idxs):
        u_i = self.U[i, :]
        m_j = self.M[j, :]
        c_vecs = [(self.Cs[ci][k, :], 0) for ci, k in enumerate(c_idxs)]
        return self.rec_dot([(u_i, 0), (m_j, 0)] + c_vecs, self.S)
        
    def loss(self, i, j, c_idxs):
        return 0.5*(self.F(i, j, c_idxs) - self.y(i, j, c_idxs))**2
        
    def df_loss(self, i, j, c_idxs):
        return self.F(i, j, c_idxs) - self.y(i, j, c_idxs)
        
    def dU(self, i, j, c_idxs):
        m_j = self.M[j, :]
        c_vecs = [(self.Cs[ci][k, :], 1) for ci, k in enumerate(c_idxs)]
        return self.rec_dot([(m_j, 1)] + c_vecs, self.S)
    
    def dM(self, i, j, c_idxs):
        u_i = self.U[i, :]
        c_vecs = [(self.Cs[ci][k, :], 1) for ci, k in enumerate(c_idxs)]
        return self.rec_dot([(u_i, 0)] + c_vecs, self.S)
    
    def dC(self, i, j, wh_c, c_idxs):
        u_i = self.U[i, :]
        m_j = self.M[j, :]
        c_vecs = [(self.Cs[ci][k, :], int(ci > wh_c)) for ci, k in enumerate(c_idxs) if ci != wh_c]
        return self.rec_dot([(u_i, 0), (m_j, 0)] + c_vecs, self.S)
    
    def rec_tensr(self, queue, out):
        if len(queue)==0: return out
        a = queue.pop(0)
        return self.rec_tensr(queue, np.multiply.outer(out, a))
    
    def dS(self, i, j, c_idxs):
        u_i = self.U[i, :]
        m_j = self.M[j, :]
        c_vecs = [self.Cs[ci][k, :] for ci, k in enumerate(c_idxs)]
        return self.rec_tensr(c_vecs, np.outer(u_i, m_j))
    
    def compute_grads(self, i, j, c_idxs):
        a = [self.dC(i,j,wh_c,c_idxs) for wh_c in range(len(c_idxs))]
        return [self.dU(i,j,c_idxs), self.dM(i,j,c_idxs), self.dS(i,j,c_idxs)] + a
    
    def tr_te_split(self):
        pos_idxs = set(list(np.random.choice(range(self.n_pos), self.n_pos//5, replace=False)))
        tr_pos_idxs = set(range(self.n_pos)) - pos_idxs
        neg_idxs = set(list(np.random.choice(range(self.n_neg), self.n_neg//5, replace=False)))
        tr_neg_idxs = set(range(self.n_neg)) - neg_idxs
        self.train_idxs = {0: list(tr_neg_idxs), 1: list(tr_pos_idxs)}
        self.test_idxs = {0: list(neg_idxs), 1: list(pos_idxs)}
        
        self.n_tr = {0: len(self.train_idxs[0]), 1:len(self.train_idxs[1])}
        self.n_te = {0: len(self.test_idxs[0]), 1:len(self.test_idxs[1])}
    
    def train(self):
        train_idxs, test_idxs, n_tr, n_te = self.tr_te_split()
        
        #alpha = 0.005
        print_time = 1000
        losses = []
        for t in range(1, int(5e6)):
        #    alpha = 1/np.sqrt(1+t)
            alpha = min(1, int(5e4)/t)
            # select i, j, k
            choice, tot = (1, self.n_pos) if np.random.random() < 0.2 else (0, self.n_neg) # do positive example:
            indices = self.Y_coords[choice][train_idxs[choice][np.random.randint(n_tr[choice])]]
            [i, j], c_idxs = indices[:2], list(indices[2:])
            
#            Y_ijk = self.y(i, j, c_idxs[::])
            F_loss = self.df_loss(i, j, c_idxs[::])
            grads = self.compute_grads(i, j, c_idxs[::])
            [d_U, d_M, d_S], dCs = grads[:3], grads[3:]
            
            self.U[i, :] -= alpha * d_U * F_loss
            self.M[j, :] -= alpha * d_M * F_loss
            for ii, k in enumerate(c_idxs):
                self.Cs[ii][k, :] -= alpha * dCs[ii] * F_loss
            self.S -= alpha * d_S * F_loss
            
            losses.append( F_loss )
            if t%print_time==0:
                print('time: '+str(t)+', loss, '+str(sum(np.abs(np.array(losses)))/len(losses)))
                losses = []
                if t%(print_time*10)==0: self.save()
    
    def feedback(self, user, context, review):
        ''' train the model on the provided example:
            - add the user profile to U
#            - record the true values to the gui database
        '''
        itms = list(review.keys())
        temp_y = {j:r for j, r in review.items()}
#        # add to user_likes!
#        c1, c2 = str(context[0]), str(context[1])
#        liked = [j for j, r in review.items() if r==1]
#        if c1 in self.user_likes[str(user)]: 
#            if c2 in self.user_likes[str(user)][c1]:
#                self.user_likes[str(user)][c1][c2] += liked
#        else:
#            self.user_likes[str(user)][c1] = {c2:liked}
        
        
        for j in itms:
            for t in range(1, 20):
                alpha = min(0.1, 10/t)
                F_loss = self.F(user, j, context) - temp_y[j]
                d_U = self.dU(user, j, context)
#                d_M = self.dM(user, j, context)
                self.U[user, :] -= alpha * d_U * F_loss
#                self.M[j, :] -= alpha * d_M * F_loss
        self.save()

    def recommend(self, user, obs_items, context, breakdown, newUser=False):
        ''' 
        data required: 
                dict: item -> popularity_normalised, dict: user -> unobserved items (observed in a test set)            
        predictions = generate r_ui for all i
        filter out already observed ones?
        
        compose final recommendation list based on:
            predicted ratings, popularity, diverse items 
        '''
        if newUser:
            # add the user...
            self.U = np.vstack((self.U, self.mtx(1, self.du)))
            p_pop = self.item_popularity / sum(self.item_popularity)
#            self.user_likes[str(user)] = {str(context[0]):{str(context[1]):[]}}
            p_unpop = 1 - p_pop
            p_unpop /= sum(p_unpop)
            pop = np.random.choice(range(self.m), 7, p=p_pop, replace=False)
            novel = np.random.choice(range(self.m), 3, p=p_unpop, replace=False)
            return [(p, 'users in similar contexts liked this', 'R') for p in pop]+[(p, 'this song hasn\'t got many ratings', 'N') for p in novel]
        
        un_obs_items = list(set(list(range(self.m))) - set(obs_items))
        selection = np.random.choice(obs_items, min(10, len(obs_items)), replace=False)
        base_Y = np.append(un_obs_items, selection)
        ''' quality is a function of user-context-item '''
        print('Producing data for DPP...')
        quality = np.array([self.F(user, j, context) * self.item_popularity[j] for j in base_Y])
        mask = quality > 0# remove any with q < 0
        quality = quality[mask]
        ''' quality is a function of the latent space of items '''
        diversity = np.array([quality[i] * self.M[j, :]/np.linalg.norm(self.M[j, :]) for i, j in enumerate(base_Y[mask])])
        # --> sample dual dpp
        C = np.dot(diversity.T, diversity)
        print('Sampling DPP...')
        dpp = DualDPP(C, diversity.T)
        selected_items = list( dpp.sample_dual(k=10) )
        print("Done.")
        for i in range(len(selected_items)):
            if selected_items[i] in selection:
                selected_items[i] = (selected_items[i], "we thought you might like to listen to this again", "L")
            else:
                selected_items[i] = (selected_items[i], "we think you'll like this based on\nyou're previous likes and context", "L")
        return selected_items

#rs = TF()
#print(rs.recommend())