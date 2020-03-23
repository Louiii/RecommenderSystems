#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:06:41 2020

@author: louisrobinson
"""
import json
from tqdm import tqdm
#from geopy import geocoders
#
#g = geocoders.GoogleV3(api_key='AIzaSyBdc089licwo8QrqLlRkIaIi6KRQpKfZLI')
#g.geocode('America/Detroit')
time_zones = {'':0,'Atlantic Time (Canada)':-3,'Mid-Atlantic':-2,'Ekaterinburg':5,'Bratislava':1,'Central America':-6,
'Astana':6,'Singapore':8,'Newfoundland':-2.5,'Perth':8,'America/New_York':-4,'America/Chicago':-5,
'America/Detroit':-4,'Kuwait':3,'Vienna':1,'Midway Island':-11,'Cairo':2,'Brussels':1,'Tokyo':9,'Alaska':-8,
'Berlin':1,'Melbourne':11,'Zagreb':1,'Bogota':-5,'Riyadh':3,'Mumbai':5.5,'Kathmandu':5.75,'New Delhi':5.5,
 'Arizona':-7,'Brasilia':-3,'Kyiv':2,'Dublin':0,'West Central Africa':1,'America/Sao_Paulo':-3,
 'Islamabad':5,'Minsk':3,'Jakarta':7,'Warsaw':1,'Muscat':4,'Copenhagen':1,'Quito':-5,'Lisbon':0,
 'Seoul':9,'Amsterdam':0,'Moscow':3,'Ljubljana':1,'Chennai':5.5,'Riga':2,'Baghdad':3,'Hawaii':-10,
 'Indiana (East)':-5,'Almaty':6,'Europe/London':0,'Pacific/Guam':10,'Osaka':9,'Hong Kong':8,
 'Tehran':3.5,'Sofia':2,'Edinburgh':0,'Sydney':11,'Europe/Minsk':3,'Lima':-5,'Chihuahua':-7,'Azores':-1,
 'Vilnius':2,'Sapporo':9,'Taipei':8,'Paris':1,'Helsinki':2,'Stockholm':1,'Mexico City':-6,
 'Saskatchewan':-6,'Santiago':-3,'Greenland':-3,'Bern':1,'International Date Line West':-12,
 'Prague':1,'Athens':2,'Belgrade':1,'Auckland':13,'Hobart':11,'Central Time (US & Canada)':-5,'Beijing':8,
 'Pretoria':2,'Adelaide':10.5,'Yakutsk':9,'Wellington':13,'Rome':1,'Abu Dhabi':4,'Novosibirsk':7,
 'London':0,'Caracas':-4,'Jerusalem':2,'Canberra':11,'Monterrey':-6,'Irkutsk':8,'Budapest':1,
 'Kuala Lumpur':8,'Istanbul':3,'Madrid':1,'Eastern Time (US & Canada)':-5,'Casablanca':1,'Guadalajara':-6,
 'Monrovia':0,'Pacific Time (US & Canada)':-8,'La Paz':-4,'St. Petersburg':3,'Tbilisi':4,'Buenos Aires':-3,
 'Mountain Time (US & Canada)':-7,'Brisbane':10,'Harare':2,'Bangkok':7,'New Caledonia':11, 'Karachi':5,
 'Urumqi':8, 'Yerevan':4, 'Tijuana':-7, 'Sri Jayawardenepura':5.5, 'Tashkent':5, 'Bucharest':2, 'Nairobi':3}



names = [("data/Context_POP_RND/train_final_POP_RND.txt", "data/rdn_train_final.json"),
        ("data/Context_POP_RND/test_final_POP_RND.txt", "data/rdn_test_final.json"),
        ("data/Context_POP_USER/train_final_POP_USER.txt", "data/usr_train_final.json"),
        ("data/Context_POP_USER/test_final_POP_USER.txt", "data/usr_test_final.json")]

#def make_final_dataset(names):
''' user, item, month, hour, time_zone, ranking
--> currently using time_zone as location 
-> if there is time add longitude
'''
users, items, locs, l_track, track_ids_map = dict(), dict(), dict(), dict(), dict()
u_idx, i_idx = 0, 0
d = []
# format columns: user_id, item_id, time_zone, time, rating
for filename, _ in tqdm(names):
    data = []
    with open(filename) as f:
        for line in f:
            ''' user_id,track_id,hashtag,created_at,score,lang,tweet_lang,
            time_zone,instrumentalness,liveness,speechiness,danceability,
            valence,loudness,tempo,acousticness,energy,mode,key,rating '''
            user_id,track_id,_,created_at,_,_,_,time_zone,_,_,_,_,_,_,_,_,_,_,_,rating = line.split('\t')
            print(track_id)
            u_id, t_id, location, r = str(user_id), str(track_id), str(time_zone), float(rating)
            time_shift = time_zones[time_zone]
            yr, mn, hr = int(created_at[:4])-2014, int(created_at[5:7]), int(created_at[11:13])
            mn += 12*yr
            hr = (hr-time_shift)%24
            if u_id not in users: 
                users[u_id] = u_idx
                u_idx += 1
            if t_id not in items: 
                items[t_id] = i_idx
                track_ids_map[i_idx] = t_id
                i_idx += 1
            if time_shift not in locs:
                locs[time_shift] = {users[u_id]: 1}
                l_track[time_shift] = [items[t_id]]
            else:
                l_track[time_shift].append(items[t_id])
                if users[u_id] not in locs[time_shift]:
                    locs[time_shift][users[u_id]] = 1
                else:
                    locs[time_shift][users[u_id]] += 1
            data.append([users[u_id], items[t_id], mn, hr, time_shift, r])
    d.append(data)

import numpy as np

# select subset of users and items based on location (I want a variety of 
# locations in my dataset- the original data has predominantely Bejing etc.)
loc_counts = {k:(len(v), sum([vi for ki, vi in v.items()])) for k, v in locs.items()}

final_users, subset_items = set(), set()
for loc, (nu, ne) in loc_counts.items():
    print((loc, (nu, ne)))
    users_at_loc, tracks_at_loc = locs[loc], list(set(l_track[loc]))
    print(tracks_at_loc)
    if ne < 500:
        for u in list(users_at_loc.keys()): final_users.add(u)
        for i in list(tracks_at_loc): subset_items.add(i)
    elif nu < 200:
        n = len(tracks_at_loc) // 2
        print((n))
        for u in list(users_at_loc.keys()): final_users.add(u)
        for i in list(np.random.choice(tracks_at_loc, n, replace=False)): subset_items.add(i)
    else:
        n = len(tracks_at_loc) // 5
        m = len(list(users_at_loc.keys())) // 5
        print((n, m))
        for u in list(np.random.choice(list(users_at_loc.keys()), m, replace=False)): 
            final_users.add(u)
        for i in list(np.random.choice(tracks_at_loc, n, replace=False)): 
            subset_items.add(i)

#
#tracks_to_include = set()
#ls = {}
#for loc, user_counts in locs.items():
#    loc_users = [k for k, v in sorted(user_counts.items(), key=lambda u_c: u_c[1])]
#    ls[loc] = (loc_users[::-1], set(l_track[loc]))
#final_users = set()
#for l,(u,c) in ls.items():
#    if len(u) < 100:
#        for t in c: tracks_to_include.add(t)
#    for ui in u[:min(len(u), 100)]:
#        final_users.add(ui)
#        
#possible_additions = set(list(items.values())) - tracks_to_include
#import numpy as np
## make a subset of items
#subs = set(np.random.choice(list(possible_additions), 3*(len(possible_additions)//10)))
#for t in tracks_to_include: subs.add(t)
#subset_items = set(list(np.array(list(items.values()))[list(subs)]))

# reindex users and items
all_users, all_items, final_track_map = dict(), dict(), dict()
u_idx, i_idx = 0, 0
d2 = []
for di in d:
    di2 = []
    for [u, i, mn, hr, l, r] in tqdm(di):
        if u in final_users:
            if i in subset_items:
                if u not in all_users: 
                    all_users[u] = u_idx
                    u_idx += 1
                if i not in all_items: 
                    all_items[i] = i_idx
                    final_track_map[i_idx] = track_ids_map[i]
                    i_idx += 1
                di2.append([all_users[u], all_items[i], mn, hr, l, r])
    d2.append(di2)
    
    
with open('my_id_to_track_id.json', 'w') as outfile:
    outfile.write(json.dumps(final_track_map))

for i, (_, fwrite) in enumerate(names):
    with open(fwrite, 'w') as outfile:
        outfile.write(json.dumps(d2[i]))
#    return len(all_users), len(all_items)
nu, ni = len(all_users), len(all_items)
    


#nu, ni = make_final_dataset(names)

