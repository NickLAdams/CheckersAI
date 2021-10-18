#!/usr/bin/env python
from __future__ import print_function

import argparse
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
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
 
GAME = 'xo' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 9 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 32. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
progressFile = 'progress.csv'
report_interval = 100

img_rows , img_cols = 3, 3
#Convert image into Black and white
img_channels = 2 #We stack 4 frames

class xoxgame:
    def __init__(self):
        self.cleargame()
        
    def cleargame(self):
        self.board = np.zeros(9)
        self.freespace = np.ones(9)
        self.player = 1
               
    def currentplayer(self):
        return(self.player)
        
    def changeplayer(self):
        self.player = self.player * (-1)
        
    def state(self):
        return(self.board.reshape(1,3,3) * self.player)
        
    def state2(self):
        st = np.stack((self.board.reshape(3,3) * self.player, self.freespace.reshape(3,3)), axis = 0)
        return(st.reshape(1,2,3,3))
        
    def freesquares(self):
        return(self.freespace)
        
    def play(self, loc):
        if self.board[loc] == 0:
            self.board[loc] = self.player
            self.freespace[loc] = 0
            return(0)
        else:
            print("loc ", loc, " already occupied")
            return(1)
            
    def won(self): # 0 = ongoing, 1 = 1 won, -1 = -1 won, 2 = draw
        if np.sum(self.freespace) > 5: # 0 if ongoing else 1, r_current player, r_opponent
            return(0, 0, 0)
        bd = self.board.reshape(3,3)
        line_total = np.sum(bd, axis = 0)
        if (self.player * 3) in line_total:
            return(1, 1, -1)
        line_total = np.sum(bd, axis = 1)
        if (self.player * 3) in line_total:
            return(1, 1, -1)
        if (self.player * 3) == (bd[0][0] + bd[1][1] + bd[2][2]):
            return(1, 1, -1)
        if (self.player * 3) == (bd[0][2] + bd[1][1] + bd[2][0]):
            return(1, 1, -1)
        if np.sum(self.freespace) == 0:
            return(1, 0.5, 0.5)
        return(0, 0, 0)
          
def chooseaction(game_state, model, board_state, choose_random):
    
    if choose_random:
        valid_places = game_state.freesquares()
        valid_places = np.nonzero(valid_places)
        action_index = random.choice(valid_places[0])
    else:
        q = model.predict(board_state)
        q = q.reshape([9])
        valid_places = game_state.freesquares()
        valid_places_index = np.nonzero(valid_places) # look at valid positions
        max_Q_valid = np.argmax(q[valid_places_index[0]]) # find max Q only in valid positions - index is for sublist
        action_index = valid_places_index[0][max_Q_valid] # convert index back to full list = action
    return(action_index)
    
    
def buildmodel():
    print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,init=lambda shape, name: normal(shape, scale=0.01, name=name), border_mode='same',input_shape=(img_channels,img_rows,img_cols)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(64, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(64, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS,init=lambda shape, name: normal(shape, scale=0.01, name=name)))
   
    adam = Adam(lr=1e-6)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = xoxgame()

    # store the previous observations in replay memory
    D = deque()
    p1_win = 0.0
    p1_lose = 0.0
    p1_draw = 0.0
    start = time.time()


    OBSERVE = OBSERVATION
    epsilon = INITIAL_EPSILON

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    

    if args['mode'] == 'Resume':                       #We go to training mode
        OBSERVE = 256    #We keep observe, never train
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=1e-6)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    

    t = 0
    with open(progressFile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(("TIMESTEP","ELAPSED","STATE","EPSILON","WINS","LOSES","DRAWS","Q_MAX","LOSS"))
        
    f.close()

    while (True):
        loss = 0
        Q_sa = 0
        action_index1 = 0
        action_index2 = 0
        r1_t = 0
        r2_t = 0
        terminal = False
        s1_t = game_state.state2()
        #choose an action epsilon greedy
        
        action_index1 = chooseaction(game_state, model, s1_t, random.random() <= epsilon)
 
        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        
        game_state.play(action_index1)
                
        game_state.changeplayer()

        s2_t = game_state.state2()
        
        while(True):
            
            action_index2 = chooseaction(game_state, model, s2_t, random.random() <= epsilon)
    
            game_state.play(action_index2)
            
            (status, r2_t, r1_t) = game_state.won()
            s2_t1 = game_state.state2()
            
            game_state.changeplayer() # here player 1 again
            
            s1_t1 = game_state.state2()
            
            if status == 1:
                terminal = True
                break
            
            # store the transition in D for player 1
            D.append((s1_t, action_index1, r1_t, s1_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            s1_t = s1_t1
    
            action_index1 = chooseaction(game_state, model, s1_t, random.random() <= epsilon)
    
            game_state.play(action_index1)
            
            (status, r1_t, r2_t) = game_state.won()
            s1_t1 = game_state.state2()
            
            game_state.changeplayer() # here player 1 again
            
            s2_t1 = game_state.state2()
            
            if status == 1:
                terminal = True
                break
            
            # store the transition in D for player 1
            D.append((s2_t, action_index2, r2_t, s2_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
    

            s2_t = s2_t1

        # we have reached the end of a game, record the terminal moves for both players
            
        # store the final score in D for player 1
        D.append((s1_t, action_index1, r1_t, s1_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # store the final score in D for player 1
        D.append((s2_t, action_index2, r2_t, s2_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if r1_t == 1:
            p1_win = p1_win + 1
        if r1_t == -1:
            p1_lose = p1_lose + 1
        if r1_t == 0.5:
            p1_draw = p1_draw + 1
            
        game_state.cleargame()
        t = t + 1

                #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, img_channels, img_rows, img_cols))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = model.predict(state_t)  # Hitting each buttom probability
                Q_sa = model.predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += model.train_on_batch(inputs, targets)

                # save progress every 10000 iterations
        if t % report_interval == 0:
            print("Now we save model")
            model.save_weights("model.h5", overwrite=True)
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
            p1_draw = p1_draw/report_interval

            with open(progressFile, 'ab') as f:
                writer = csv.writer(f)
                s = (t ,elapsed,state,epsilon,p1_win,p1_lose,p1_draw,np.max(Q_sa),loss)
                writer.writerow(s)
                
            f.close()
            print("TIMESTEP", t, " Elapsed, ", elapsed, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ WINS", p1_win, "/ LOSSES", p1_lose, \
                "/ DRAWS", p1_draw, "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
            p1_win = 0.0
            p1_lose = 0.0
            p1_draw = 0.0


    print("Episode finished!")
    print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    main()
