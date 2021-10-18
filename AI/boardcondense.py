#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 20:47:07 2017

@author: Nick Adams
"""

import numpy as np
import pickle
from collections import defaultdict



file = open("validMoves.txt","rb") # use small moves for testing
moves = pickle.load(file)
file.close()


file = open("validPos.txt","rb") # use small moves for testing
pos = pickle.load(file)
file.close()

#pos.reshape((len(pos), 128))
print('Boards to analyse', len(moves))

unique_bds = defaultdict(int)
for i in range(len(moves)):
    bd = np.asarray(moves[i, ...])+2
    key = ''.join(''.join('%0.0f' %x for x in y) for y in bd)
    if key not in unique_bds:
        unique_bds[key] = np.asarray(pos[i, ...])
    else:
        unique_bds[key] = unique_bds[key] + np.asarray(pos[i, ...])

unique_count = len(unique_bds)
bds_out = np.zeros((unique_count, 8, 8, 1))
pos_out = np.zeros((unique_count, 32, 4))
ii = 0
for k, v in unique_bds.items():
    al = np.asarray([int(i) for i in k]) - 2
    al = al.reshape(8,8,1)
    v = (v > 0) * 1
    bds_out[ii, ...] = al[:,...]
    pos_out[ii, ...] = v[:, ...]
    ii += 1
    
np.save('bds_out.npy', bds_out)   
np.save('pos_out.npy', pos_out)   
print('Unique boards', len(bds_out))

