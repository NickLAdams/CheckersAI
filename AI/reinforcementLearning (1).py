# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:02:30 2017

@author: User
"""


import sys
sys.path.append("game/")
import random
import numpy as np
from collections import deque
import time
import csv

import json
from keras import initializations
from keras.initializations import normal, identity
from keras.models import model_from_json , load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

GAME = 'checkers' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 128 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 32. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
progressFile = 'progress.csv'
report_interval = 100

img_rows , img_cols = 8, 8
#Convert image into Black and white
img_channels = 1 #We stack 4 frames

def trainNetwork(model):
    # open up a game state to communicate with emulator
    game_state = checkers()

    # store the previous observations in replay memory
    D = deque()
    p1_win = 0.0
    p1_lose = 0.0
    start = time.time()


    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON


    OBSERVE = 256    
    print ("Now we load weight")
    model.load_weights("weights_1024N.h5")
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print ("Weight load successfully")    

#    if args['mode'] == 'Resume':                       #We go to training mode
#        OBSERVE = 256    #We keep observe, never train
#        print ("Now we load weight")
#        model.load_weights("model.h5")
#        adam = Adam(lr=1e-6)
#        model.compile(loss='mse',optimizer=adam)
#        print ("Weight load successfully")    

    t = 0
    with open(progressFile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("TIMESTEP","ELAPSED","STATE","EPSILON","WINS","LOSES","DRAWS","Q_MAX","LOSS"))
        
    f.close()

    while (True):
        
        learning_loss = 0
        Q_sa = 0
        action_index1 = 0
        action_index2 = 0
        r1_t1 = 0
        r2_t1 = 0
        r1_t2 = 0
        r2_t2 = 0
        terminal = False
        s1_t = game_state.getBoard()
        #choose an action epsilon greedy
        
        action_index1 , game_loss = chooseaction(game_state, model, s1_t, random.random() <= epsilon)
        #print(action_index1)
        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        
        game_state.makeMove(action_index1[0],action_index1[1],False)
        #input()
                
        #game_state.changeplayer()

        s2_t32, s2_t = game_state.flipBoard()
        
        while(True):
            game_loss = False
            #game_state.printBoard()
            #input()
            
            action_index2 , game_loss = chooseaction(game_state, model, s2_t, True)
            #print(action_index2)
            
            if game_loss == False:
                game_state.makeMove(action_index2[0],action_index2[1],False)
                (status, r2_t1, r1_t1) = game_state.won()
                winner = False
                s2_t1 = game_state.getBoard()
                s1_t32 , s1_t1 = game_state.flipBoard()
                valid_places = game_state.validMoves()
                valid_places = np.nonzero(valid_places)
                if valid_places[0].size == 0:
                    status = 1
                    r1_t = -10
                    r2_t = 10
                    winner = False
                
            
            #input()
            
            #game_state.changeplayer() # here player 1 again
            
                
            if game_loss == True:
                status = 1
                r2_t = -10
                r1_t = 10
                winner = True
            
            if status == 1:
                
                terminal = True
                break
            
            r1_t = r1_t1 + r1_t2
            D.append((s1_t, action_index1, r1_t, s1_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            s1_t = s1_t1
    
            action_index1, game_loss = chooseaction(game_state, model, s1_t, random.random() <= epsilon)
            #print(action_index1)
            if game_loss == False:
                game_state.makeMove(action_index1[0],action_index1[1],False)
            
                (status, r1_t2, r2_t2) = game_state.won()
                s1_t1 = game_state.getBoard()
                winner = True
                s2_t32 ,s2_t1 = game_state.flipBoard()
                valid_places = game_state.validMoves()
                valid_places = np.nonzero(valid_places)
                if valid_places[0].size == 0:
                    status = 1
                    r1_t = 10
                    r2_t = -10
                    winner = True
                    
                    
                
            #game_state.changeplayer() # here player 1 again
            
            #input()
            
                
            if game_loss == True:
                
                status = 1
                r1_t = -10
                r2_t = 10
                winner = False
            
            if status == 1:
                
                terminal = True
                break
            
            # store the transition in D for player 1
            r2_t = r2_t1 + r2_t2
            D.append((s2_t, action_index2, r2_t, s2_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
    

            s2_t = s2_t1

        # we have reached the end of a game, record the terminal moves for both players
        # store the final score in D for player 1
        D.append((s1_t, action_index1, r1_t, s1_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # store the final score in D for player 2
        D.append((s2_t, action_index2, r2_t, s2_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if winner == True:
            p1_win = p1_win + 1
        
        if winner == False:
            p1_lose = p1_lose + 1
        
            
        game_state.reset()
        t = t + 1

                #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, img_rows, img_cols, img_channels))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0],32,4))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward
                state_t = np.reshape(state_t,(1,8,8,1))
                inputs[i:i + 1] = state_t    #I saved down s_t
                targets[i] = model.predict(state_t)# Hitting each buttom probability
                
                state_t1 = np.reshape(state_t1,(1,8,8,1))
                Q_sa = model.predict(state_t1)
                #print("action_t",action_t)
                
                if terminal:
                    targets[i][action_t] = reward_t
                else:
                    targets[i][action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            learning_loss += model.train_on_batch(inputs, targets)

                # save progress every 10000 iterations
        if t % report_interval == 0:
            print("Now we save model")
            model.save_weights("weights_1024N.h5", overwrite=True)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"
    
            done = time.time()
            elapsed = round(done - start)
            p1_win = p1_win/report_interval
            p1_lose = p1_lose/report_interval
            

            with open(progressFile, 'a') as f:
                writer = csv.writer(f)
                s = (t ,elapsed,state,epsilon,p1_win,p1_lose,np.max(Q_sa),learning_loss)
                writer.writerow(s)
                
            f.close()
            print("TIMESTEP", t, " Elapsed, ", elapsed, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ WINS", p1_win, "/ LOSSES", p1_lose, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", learning_loss)
            p1_win = 0.0
            p1_lose = 0.0
            


    print("Episode finished!")
    print("************************")
    
def chooseaction(game_state, model, board_state, choose_random):
    game_loss = False
    if choose_random:
        valid_places = game_state.validMoves()
        valid_places = np.nonzero(valid_places)
        if valid_places[0].size == 0:
            game_loss = True
            action_index = [0,3]
        else:
        #print(valid_places)
        #game_state.printBoard()
            index = random.randint(0,len(valid_places[0])-1)
            action_index = [valid_places[0][index],valid_places[1][index]]
    else:
        board_state = np.reshape(board_state,(1,8,8,1))
        #print(board_state)
        q = model.predict(board_state)
        q = q.reshape((32,4))
        #print(q)
        valid_places = game_state.validMoves()
        valid_places_index = np.nonzero(valid_places)# look at valid positions
        if valid_places_index[0].size == 0:
            game_loss = True
            action_index = [0,3]
        else:    
        #print(valid_places_index)
        #game_state.printBoard()
            max_Q_valid = np.argmax(q[valid_places_index]) # find max Q only in valid positions - index is for sublist
            action_index = [valid_places_index[0][max_Q_valid],valid_places_index[1][max_Q_valid]] # convert index back to full list = action
    return action_index , game_loss
    
class checkers:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.board64 = np.array([[0,-1,0,-1,0,-1,0,-1],[-1,0,-1,0,-1,0,-1,0],[0,-1,0,-1,0,-1,0,-1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]])
        self.board32 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        
    def printBoard(self):
        print(self.board32)
        for i in range(8):
            print(self.board64[i])
    def findValues(self,counter,direction):
        #self.counter = np.argmax(counter) 
        self.counter = counter
        self.direction = direction
        #print("directions:",direction)
        #print(self.counter)
        
        if self.counter in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
            if self.direction == 0:
                self.jump = 5
            if self.direction == 1:
                self.jump = 4
            if self.direction == 2:
                self.jump = -3
            if self.direction == 3:
                self.jump = -4
        
        if self.counter in [4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]:
            if self.direction == 0:
                self.jump = 4
            if self.direction == 1:
                self.jump = 3
            if self.direction == 2:
                self.jump = -4
            if self.direction == 3:
                self.jump = -5
        return self.counter, self.jump 
    
    def flipBoard(self):
        tempBoard = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        #print("board")
        #print(board)
        for i in range(32):
            tempBoard[i] = -(self.board32[31-i])
            
        
        temp64 = np.zeros(64)
        for x in range(32):
            if x in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
                temp64[2*x+1] = tempBoard[x]
            else:
                temp64[2*x] = tempBoard[x]
                
        temp64 = np.reshape(temp64,(8,8))
        self.board32 = tempBoard
        self.board64 = temp64
        return tempBoard, temp64
    
    def validMoves(self):
        self.validList = np.zeros((32,4))
        kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]],
                    4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]],
                    8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]],
                    11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]],15 : [[10,6],[18,22]],
                    16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]],
                    20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 
                    24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 
                    28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
        jump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
        kingDir = {0 : [0], 1 : [1,0], 2 : [1,0], 3 : [1], 
                   4 : [0], 5 : [1,0], 6 : [1,0], 7 : [1],
                     8 :[0,2], 9 : [3,2,1,0], 10 : [3,2,1,0], 
                      11 : [3,1], 12 : [2,0], 13 : [3,2,1,0], 14 : [3,2,1,0], 15 : [3,1], 
                        16 : [2,0], 17 : [3,2,1,0], 18 : [3,2,1,0], 19 : [3,1], 
                        20 : [2,0], 21 : [3,2,1,0], 22 : [3,2,1,0], 23 : [3,1], 
                        24 : [2], 25 : [3,2], 26 : [3,2],27 : [3], 
                        28 : [2], 29 : [3,2], 30 : [3,2], 31 : [3]}
        Dir = {0 : [0], 1 : [1,0], 2 : [1,0], 3 : [1], 
               4 : [0], 5 : [1,0], 6 : [1,0], 7 : [1], 
               8 :[0], 9 : [1,0], 10 : [1,0], 11 : [1], 
                  12 : [0], 13 : [1,0], 14 : [1,0], 15 : [1], 
                       16 : [0], 17 : [1,0], 18 : [1,0], 19 : [1], 
                            20 : [0], 21 : [1,0], 22 : [1,0]}
        step = {0: [4], 7:[11],8:[12],15:[19],16:[20],23:[27], 24 : [28,29], 25 : [29,30], 26 : [30,31], 27:[31]}
        stepDir = {0: [1], 7 : [0], 8: [1] , 15: [0], 16 : [1], 23 : [1],24 :[1,0], 25: [1,0],26:[1,0],27:[1]}
        kingStep = {4 : [0], 5:[0,1], 6:[1,2],7:[2,3,11],0: [4],8:[4,12],15:[11,19],16:[12,20],23:[19,27],24 : [28,29], 25 : [29,30], 26 : [30,31], 27:[31], 31 : [26,27]}
        kingStepDir = {4 : [2], 5:[3,2],6:[3,2],7:[3,2,0],0: [1], 8: [3,1] , 15: [2,0], 16 : [3,1], 23 : [2,1],24 :[1,0], 25: [1,0],26:[1,0],27:[1], 31 : [3,2]}
        jumps = 0
        for i in range(32):
            if self.board32[i] == -1 and i < 23:
                for x in range(len(jump[i])):
                    if self.board32[jump[i][x][0]] > 0 and self.board32[jump[i][x][1]] == 0:
                        self.validList[i][Dir[i][x]] = 1
                        jumps += 1
            if self.board32[i] == -2:
                for x in range(len(kingJump[i])):
                    if self.board32[kingJump[i][x][0]] > 0 and self.board32[kingJump[i][x][1]] == 0:
                        self.validList[i][kingDir[i][x]] = 1
                        jumps += 1
                    
        if jumps == 0:
            #print("no jumps")
            for i in range(32):
                if self.board32[i] == -1 and i < 23:
                    for x in range(len(jump[i])):
                        if self.board32[jump[i][x][0]] == 0:
                            self.validList[i][Dir[i][x]] = 1
                                          
                if self.board32[i] == -1 and i < 28 and i > 23:
                    for x in range(len(step[i])):
                        if self.board32[step[i][x]] == 0:
                            self.validList[i][stepDir[i][x]] = 1
                
                if self.board32[i] == -1 and i in [0,7,8,15,16,23]:
                    for x in range(len(step[i])):
                        if self.board32[step[i][x]] == 0:
                            self.validList[i][stepDir[i][x]] = 1
                                          
                if self.board32[i] == -2:
                    for x in range(len(kingJump[i])):
                        if self.board32[kingJump[i][x][0]] == 0:
                            self.validList[i][kingDir[i][x]] = 1
                                              
                    if i in [0,4,7,8,15,16,23,24,25,26,27,31]:
                        for x in range(len(kingStep[i])):
                            if self.board32[kingStep[i][x]] == 0:
                                self.validList[i][kingStepDir[i][x]] = 1
                        
        #print(self.validList)                           
        return self.validList
    
    
    
    def makeMove(self,counter,direction,jump):
        if jump == False:
            self.points = 0
        counter , jump = self.findValues(counter,direction)
        #print(counter,jump)
        kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
        normalJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
        
        anotherJump = False
        
        endPos = counter + jump
        piece = self.board32[counter]
        if self.board32[endPos] == 0:
            self.board32[counter] = 0
            if endPos > 27:
                piece = -2
                self.points += 1
            self.board32[endPos] = piece
            #print("1")
        else:
            if piece == -1:
                for i in range(len(normalJump[counter])):
                    if normalJump[counter][i][0] == endPos:
                        self.board32[counter] = 0
                        self.board32[endPos] = 0
                        if normalJump[counter][i][1] > 27:
                            piece = -2
                            self.points += 1
                        self.board32[normalJump[counter][i][1]] = piece
                        self.points += 1
                        #print("2")           
                        anotherJump = self.checkJump(normalJump[counter][i][1])
            else:
                for i in range(len(kingJump[counter])):
                    if kingJump[counter][i][0] == endPos:
                        self.board32[counter] = 0
                        self.board32[endPos] = 0
                        self.board32[kingJump[counter][i][1]] = piece
                        self.points += 1
                        #print("3")
                        anotherJump , self.validList = self.checkJump(kingJump[counter][i][1])
        self.update64()
        if anotherJump == True:
            nextMove = np.unravel_index(np.argmax(self.validList),self.validList.shape)
            self.makeMove(int(nextMove[0]),int(nextMove[1]),True)
            
        
        
        #return anotherJump, self.validList
                        
    def checkJump(self,counter):
        kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
        jump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
        kingDir = {0 : [0], 1 : [1,0], 2 : [1,0], 3 : [1], 
                   4 : [0], 5 : [1,0], 6 : [1,0], 7 : [1],
                     8 :[0,2], 9 : [3,2,1,0], 10 : [3,2,1,0], 
                      11 : [3,1], 12 : [2,0], 13 : [3,2,1,0], 14 : [3,2,1,0], 15 : [3,1], 
                        16 : [2,0], 17 : [3,2,1,0], 18 : [3,2,1,0], 19 : [3,1], 
                        20 : [2,0], 21 : [3,2,1,0], 22 : [3,2,1,0], 23 : [3,1], 
                        24 : [2], 25 : [3,2], 26 : [3,2],27 : [3], 
                        28 : [2], 29 : [3,2], 30 : [3,2], 31 : [3]}
        Dir = {0 : [0], 1 : [1,0], 2 : [1,0], 3 : [1], 
               4 : [0], 5 : [1,0], 6 : [1,0], 7 : [1], 
               8 :[0], 9 : [1,0], 10 : [1,0], 11 : [1], 
                  12 : [0], 13 : [1,0], 14 : [1,0], 15 : [1], 
                       16 : [0], 17 : [1,0], 18 : [1,0], 19 : [1], 
                            20 : [0], 21 : [1,0], 22 : [1,0], 23 : [1]}
        
        self.validList = np.zeros((32,4))
        
        anotherJump = False
        if self.board32[counter] == -1 and counter < 24:
            for x in range(len(jump[counter])):
                if self.board32[jump[counter][x][0]] > 0 and self.board32[jump[counter][x][1]] == 0:
                    anotherJump = True
                    self.validList[counter][Dir[counter][x]] = 1
                    
        if self.board32[counter] == -2:
            for x in range(len(kingJump[counter])):
                if self.board32[kingJump[counter][x][0]] > 0 and self.board32[kingJump[counter][x][1]] == 0:
                    anotherJump = True
                    self.validList[counter][kingDir[counter][x]] = 1
        
        return anotherJump, self.validList
    
            
    def update64(self):
        temp64 = np.zeros(64)
        for x in range(32):
            if x in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
                temp64[2*x+1] = self.board32[x]
            else:
                temp64[2*x] = self.board32[x]
                
        temp64 = np.reshape(temp64,(8,8))
        self.board64 = temp64
        
    def checkWin(self):
        p1 = 0
        p2 = 0
        
        for i in range(32):
            if self.board32[i] < 0:
                p1 += 1
            if self.board32[i] > 0:
                p2 += 1
        if p1 < 3:
            return -1
        if p2 < 3:
            return 1
        else:
            return 0 
        
    def getBoard(self):
        return self.board64
    
    def won(self):
        endGame = self.checkWin()
        if endGame == -1:
            return (1,10,-10)
        if endGame == 1:
            return (1,-10,10)
        if endGame ==0:
            
            return (0,self.points,-self.points)
        
            
model = load_model("my_model_1024.h5")
                  
trainNetwork(model)   
        
    