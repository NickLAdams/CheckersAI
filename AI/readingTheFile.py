# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:33:47 2017

@author: User
"""


import pickle
import numpy as np



file = open("moveFile.txt","rb")
moves = pickle.load(file)
file.close()

print(moves[0])
#temp = np.zeros(64)
for i in range(len(moves)):
    temp = np.zeros(64)
    for x in range(32):
        if x in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
            temp[2*x+1] = moves[i][0][x]
        else:
            temp[2*x] = moves[i][0][x]
    moves[i][0] = temp
    moves[i][0] = np.reshape(moves[i][0],(8,8))
    #for y in range(8):
    #    print(moves[i][0][y])
    #print(moves[i][1],moves[i][2])
    
#print(moves[0][0])
file = open("convertMoves.txt","wb")
pickle.dump(moves,file)
file.close()
print("done")