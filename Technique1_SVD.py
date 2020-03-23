#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:46:24 2020

@author: louisrobinson
"""

import numpy as np
import json

def load_data():
    ''' [user, item, mn, hr, location, r] '''
    '''
    POP_RND 
    we select 9 tracks randomly that the user has not listened to
    '''
    with open("data/rdn_train_final.json", 'r') as f: rdn_train = json.load(f)
    with open("data/rdn_test_final.json", 'r') as f: rdn_test = json.load(f)
        
#    '''
#    POP_USER
#    picks 9 track that the user has previously listened to, but in a different 
#    context. Track recommendation for POP_USER is more difficult as the 
#    recommender system will have to rely on contextual features to make its 
#    decision.
#    '''
#    with open("data/usr_train_final.json", 'r') as f:
#        usr_train = json.load(f)
#    with open("data/usr_test_final.json", 'r') as f:
#        usr_test = json.load(f)
        
    return (np.array(rdn_train, dtype=int), np.array(rdn_test, dtype=int))#, 
#            np.array(usr_train, dtype=int), np.array(usr_test, dtype=int))

# user, item, mn, hr, time_shift, r
# time_shift is an approximation of location

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

def makeY(data, n, m, n_contexts=2):
    all_ = set(range(len(data)))
    pos = set(list(np.nonzero(data[:, -1])[0]))
    neg = list(all_ - pos)
    Y_coords = {0:data[neg, :4], 1:data[list(pos), :4]}
    Y_tele = dict()
    
    c1, c2 = set([]), set([])
    item_popularity = np.zeros(m)
    user_likes = {str(i):dict() for i in range(n)}
    contexts_av_rating = {'0':{str(i):dict() for i in range(24)}, '1':{str(i):dict() for i in range(12)}}
    for [i, j, k, l, r] in data:
        c1.add(k)
        c2.add(l)
        if r==1:
            if str(k) in user_likes[str(i)]:
                if str(l) in user_likes[str(i)][str(k)]:
                    user_likes[str(i)][str(k)][str(l)].add(j)
                else:
                    user_likes[str(i)][str(k)][str(l)] = set([j])
            else:
                user_likes[str(i)][str(k)] = {str(l):set([j])}
            if i in Y_tele:
                Y_tele[i].add(j)
            else:
                Y_tele[i] = set([j])
        # also compute item popularity (number of ratings)
        item_popularity[i] += 1
        if str(j) in contexts_av_rating['0'][str(k)]:
            tp = contexts_av_rating['0'][str(k)][str(j)]
            contexts_av_rating['0'][str(k)][str(j)] = (tp[0]+r, tp[1]+1)
        else:
            contexts_av_rating['0'][str(k)][str(j)] = (r, 1)
        if str(j) in contexts_av_rating['1'][str(l)]:
            tp = contexts_av_rating['1'][str(l)][str(j)]
            contexts_av_rating['1'][str(l)][str(j)] = (tp[0]+r, tp[1]+1)
        else:
            contexts_av_rating['1'][str(l)][str(j)] = (r, 1)
    for k in c1: 
        for j, v in contexts_av_rating['0'][str(k)].items():
            contexts_av_rating['0'][str(k)][str(j)] = v[0]/v[1]
    for l in c2: 
        for j, v in contexts_av_rating['1'][str(l)].items():
            contexts_av_rating['1'][str(l)][str(j)] = v[0]/v[1]
    # convert all sets in user_likes to lists so that it is json serialisable
    for i, ui in user_likes.items():
        for k, uik in ui.items():
            for l, uikl in uik.items():
                user_likes[str(i)][str(k)][str(l)] = [str(j) for j in uikl]
    return Y_tele, Y_coords, contexts_av_rating, item_popularity, user_likes, len(pos), len(neg)

class SVD:
    def __init__(self, train=False):
        self.n_contexts = 2
    
        self.mtx = lambda a, b: np.random.normal(0, 0.05, (a, b))
        
        if train:
            print('loading data...')
            rdn_tr, rdn_te = load_data()#, usr_tr, usr_te
            print('transforming data...')
        
            dataset1 = conv(rdn_tr)
            dataset2 = conv(rdn_te)
            self.dataset = np.concatenate((dataset1, dataset2), axis=0)
            
            self.du, self.dm = 15, 15# latent dimensions
            self.n = len(set(list(self.dataset[:,0])))# num users, approx 2324
            self.m = len(set(list(self.dataset[:,1])))# num tracks, approx 7886
            # ^ dims of tensor Y (only partially stored in memory).
        
            self.Y_tele, self.Y_coords, self.contexts_av_rating, self.item_popularity, self.user_likes, self.n_pos, self.n_neg = makeY(self.dataset, self.n, self.m, self.n_contexts)
            self.save_pred(no_model=True)
            
            self.tr_te_split()
            
            self.U, self.M = self.mtx(self.n, self.du), self.mtx(self.m, self.dm)
            
            self.mean = self.n_pos/(self.n_neg + self.n_pos)
            self.calc_biases()
        else:
            print('loading model...')
            self.load()
            print('done.')
            
    def user_fav_by_context(self, u, c1, c2):
        u, c1, c2 = str(u), str(c1), str(c2)
        if c1 in self.user_likes[u]:
            if c2 in self.user_likes[u][c1]:
                self.user_likes[u][c1][c2] = list(set(self.user_likes[u][c1][c2]))
                return self.user_likes[u][c1][c2]
        return []

    def rec_y(self, queue, dict_):
        if len(queue)==0: return 1
        ii = queue.pop(0)
        if ii in dict_:
            return self.rec_y(queue, dict_[ii])
        return 0

    def y(self, i, j):
        if i in self.Y_tele:
            if j in self.Y_tele[i]:
                return 1
        return 0

    def calc_biases(self):
        bu, bi, uc, ic = np.zeros(self.n), np.zeros(self.m), np.zeros(self.n), np.zeros(self.m)
        for [u, i, _, _, r] in self.dataset:
            bu[u] += r
            bi[i] += r
            uc[u] += 1
            ic[i] += 1
        self.bu, self.bi = bu/uc, bi/ic

    def save(self, path='SVD-model/'):
        np.save(path+'U', self.U), np.save(path+'M', self.M)
        np.save(path+'Bu', self.bu), np.save(path+'Bi', self.bi)
        np.save(path+'mean', self.mean)
    
    def save_pred(self, no_model=False):
        np.save('SVD-model/item_popularity', self.item_popularity)
        with open('SVD-model/contexts_av_rating.json', 'w') as f: f.write(json.dumps(self.contexts_av_rating))
        with open('SVD-model/user_likes.json', 'w') as f: f.write(json.dumps(self.user_likes))
        if not no_model: self.save()
        
    def load(self, path='SVD-model/'):
        self.item_popularity = np.load(path+'item_popularity.npy')
        with open("SVD-model/contexts_av_rating.json", 'r') as f: self.contexts_av_rating = json.load(f)
        with open("SVD-model/user_likes.json", 'r') as f: self.user_likes = json.load(f)
        self.U, self.M = np.load(path+'U.npy'), np.load(path+'M.npy')
        self.bu, self.bi = np.load(path+'Bu.npy'), np.load(path+'Bi.npy')
        self.mean = np.load(path+'mean.npy')
        (self.n, self.du), (self.m, self.dm) = self.U.shape, self.M.shape
        
    def F(self, i, j):
        u_i = self.U[i, :]
        m_j = self.M[j, :]
        return self.mean + self.bu[i] + self.bi[j] + np.dot(u_i, m_j)
        
    def loss(self, i, j):
        return 0.5*(self.F(i, j) - self.y(i, j))**2
        
    def df_loss(self, i, j):
        return self.F(i, j) - self.y(i, j)
        
    def dU(self, i, j):
        return self.M[j, :]

    def dM(self, i, j):
        return self.U[i, :]

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
        ''' training takes about 1 min per million steps, usually the loss 
        drops below 0.005 with 5M-10M steps, but step the max time high and 
        stop whenever you are satisfied with the loss as it saves itself '''
        print_time = 1000
        losses = []
        for t in range(1, int(1e7)):
            alpha = min(0.3, int(1e4)/t)
            
            choice, tot = (1, self.n_pos) if np.random.random() < 0.2 else (0, self.n_neg) # do positive example:
            indices = self.Y_coords[choice][self.train_idxs[choice][np.random.randint(self.n_tr[choice])]]
            [i, j] = indices[:2]
            
#            Y_ijk = self.y(i, j)
            F_loss = self.df_loss(i, j)
            d_U, d_M = self.dU(i, j), self.dM(i, j)
            
            self.U[i, :] -= alpha * d_U * F_loss
            self.M[j, :] -= alpha * d_M * F_loss
            
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
        
        c1, c2 = str(context[0]), str(context[1])
        liked = [j for j, r in review.items() if r==1]
        if c1 in self.user_likes[str(user)]: 
            if c2 in self.user_likes[str(user)][c1]:
                self.user_likes[str(user)][c1][c2] += liked
        else:
            self.user_likes[str(user)][c1] = {c2:liked}
        
        for j in itms:
            for t in range(1, 20):
                alpha = min(0.1, 10/t)
                F_loss = self.F(user, j) - temp_y[j]
                d_U = self.dU(user, j)
#                d_M = self.dM(user, j)
                self.U[user, :] -= alpha * d_U * F_loss
#                self.M[j, :] -= alpha * d_M * F_loss
        self.save()
            
    def find_items_u(self, user, context, n_it):
        ''' return a sorted (by similarity of context) list of items that the user has liked previously '''
        [mnth, loc_tz] = context
        mx_mnth, mx_loc = 12, 24
        
        sim_mnths = [(i, np.abs(i-mnth)) for i in range(mx_mnth)]
        sim_locs = [(i, np.abs(i-loc_tz)) for i in range(mx_loc)]
        sim = [((mnth, lc), distm+distl) for mnth, distm in sim_mnths for lc, distl in sim_locs]
        sim = sorted(sim, key=lambda x:x[1])
        queue = [a for a, b in sim]
        
        collection = []
        while len(collection) < n_it and len(queue) > 0:
            c1, c2 = queue.pop(0)
            collection += self.user_fav_by_context(str(user), str(c1), str(c2))
        return collection[:min(n_it, len(collection))]

    def recommend(self, user, obs_items, context, breakdown, newUser=False):
        ''' use SVD predicted ratings over {user} x [unobserved items] -> array of ratings
        use stored array of previous item average rating in each context , contexts_av_rating
        compute = r_hat + (avrt[c1] +...+ avrt[cn])/n
        rank
              (1) prepare top 10 ranked -> unobserved, same context
              (2) selection the user has seen (liked) in that context -> else that they have liked at all
              (3) prepare some unpopular (novel) recommendations 
        compose final list based of n from (1), (2), (3).
        '''
        print('#########'+str(context)+'#########')
        c1, c2 = str(context[0]), str(context[1])
        c1_pop, c2_pop = self.contexts_av_rating['0'][c1], self.contexts_av_rating['1'][c2]
        item_pop_dep_c = dict()
        for i in range(self.m):
            v = 0.0
            if str(i) in c1_pop: v += c1_pop[str(i)]
            if str(i) in c2_pop: v += c2_pop[str(i)]
            if v > 0: item_pop_dep_c[i] = v

        un_obs_items = list(set(list(range(self.m))) - set(obs_items))
        
        if newUser:
            # add the user...
            self.U = np.vstack((self.U, self.mtx(1, self.du)))
            self.bu = np.append(self.bu, 0)
            self.user_likes[str(user)] = {c1:{c2:[]}}
            # select the most popular items in that context.
            item_pop_dep_c = [(k, v) for k, v in item_pop_dep_c.items()]
            item_pop_dep_c = sorted(item_pop_dep_c, key=lambda x:-x[1])[:10]
            item_pop_dep_c = [i for (i, p) in item_pop_dep_c]
            # select some novel items.
            un_obs_items = [i for i in un_obs_items if i not in item_pop_dep_c]
            ui_pop = np.array([1/(1+self.item_popularity[i]) for i in un_obs_items])
            novel = np.random.choice(un_obs_items, 10, p=ui_pop/sum(ui_pop), replace=False)
            return [(item_pop_dep_c[i], 'users in similar contexts liked this', 'R') for i in range(6)]+[(novel[i], 'this song hasn\'t got many ratings', 'N') for i in range(4)]
        
        ratings_for_u = np.array([self.F(user, j) for j in un_obs_items])
        for i, j in enumerate(un_obs_items):
            if j in item_pop_dep_c:
                ratings_for_u[i] += item_pop_dep_c[j]# a factor could be used in front of this, if the user is new- less of an effect from F
        
        ranks = list(zip(un_obs_items, ratings_for_u))
        ranks = sorted(list(ranks), key=lambda x:x[1])[-min(len(ranks), 10):]
        ranks = [itm for (itm, rnk) in ranks][::-1]# (1)
        
        
        # select some novel items.
        un_obs_items = [i for i in un_obs_items if i not in ranks]
        ui_pop = np.array([1/(1+self.item_popularity[i]) for i in un_obs_items])
        novel = np.random.choice(un_obs_items, 10, p=ui_pop/sum(ui_pop), replace=False)
        novel = [(n, 'this song hasn\'t got many ratings', 'N') for n in novel]
        ranks = [(r, 'similar users in similar contexts liked this', 'R') for r in ranks]
        
        # prepare songs that the user has already liked in that context,
        already_liked = self.user_fav_by_context(str(user), c1, c2)
        already_liked = [(a, 'you liked this in this context', 'L') for a in already_liked]
        if len(already_liked)<3:
            # if none: some they have liked at all, (first search most similar contexts)
            da = self.find_items_u(user, context, 3)
            da = [(a, 'you liked this', 'L') for a in da]
            already_liked += da
        if len(already_liked)<3:
            already_liked += novel[-3:]
        
        
        if breakdown == None:# return 10!
            return ranks[:5] + already_liked[:3] + novel[:2]
        
#        novel_pref, repeat_pref, pop_pref = c['N'], breakdown['A'], breakdown['R']
        prefs = [(p, r) for p, r in breakdown.items()]
        n_br = sum([r for (p, r) in prefs])
        prefs = sorted(prefs, key=lambda x:x[1])
        
        if n_br==0: return ranks[:5] + already_liked[:3] + novel[:2]
        chosen = prefs[-1][0]
        if chosen=='N':
            if n_br in [1, 2]: return ranks[:2] + novel[:3]
            return ranks[:2] + already_liked[:1] + novel[:7]
        if chosen=='R':
            if n_br in [1, 2]: return ranks[:3] + already_liked[:1] + novel[:1]
            return ranks[:7] + already_liked[:2] + novel[:1]
        if n_br in [1, 2]: return ranks[:1] + already_liked[:3] + novel[:1]
        return ranks[:3] + already_liked[:6] + novel[:1]
#rs = SVD(train=True)
#rs.train()
            
#rs = SVD()
