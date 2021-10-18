# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:29:31 2017

@author: User
"""

##Save current board + the move that will happen + flip board and move 

# error on line 734

#import the file reading code
import readGameFile as readGame
import numpy as np
import pickle

def pos(counter):
    pos = {0: (4,5),1:(5,6),2:(6,7),3:7,4:8,5:(8,9),6:(9,10),7:(10,11),8:(12,13),9:(13,14),10:(14,15),11:15,12:16,13:(16,17),14:(17,18),15:(18,19),16:(20,21),17:(21,22),18:(22,23),19:23,20:24,21:(24,25),22:(25,26),23:(26,27),24:(28,29),25:(29,30),26:(30,31),27:31,28:-1,29:-1,30:-1,31:-1}
    return pos[counter]

def counterMoves(i,counter,gameCurrent,player):
    #print(i)
    #print("Counter")
    #print(counter)
    foundStart = False
    foundEnd = False
    end = 0
    start = 0
    
    
    
    if player == True:
        counter = 32 - counter
        #print("Counter converted")
        #print(counter)

    if i % 2 == 0:
        for x in range(i+1, len(gameCurrent)):
            
            if int(gameCurrent[x][0])-1 == counter and foundStart == False:
                #print("StartPos")
                #print(x)
                start = x
                foundStart = True
            
            if int(gameCurrent[x][2])-1 == counter and foundEnd == False:
                #print("EndPos")
                #print(x)
                foundEnd = True
                end = x
    else:
        for x in range(i+1, len(gameCurrent)):
            #print("gameCurrent")
            #print(gameCurrent[x])
            
            
            if int(gameCurrent[x][0]) == counter and foundStart == False:
                #print("StartPos")
                #print(x)
                start = x
                foundStart = True
            
            if int(gameCurrent[x][2]) == counter and foundEnd == False:
               # print("EndPos")
                #print(x)
                end = x
                foundEnd = True
                
    
    if end < start:
        return False
    if start < end and start != 0:
        print("Failed checking for movement")
        return True
    else:
        return False

def getReadGame(needNew):
    needNew = True
    #creates a formatted array
    gameBoards = np.array([[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],[0],[0,0,0,0],[1]]])
    #sets first value to starting state
    jump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
    jumpKing = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}

    board = np.array([[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],[0],[0,0,0,0],[1]]])


    if needNew == True:
        #gets list of game moves
        gameCurrent = readGame.readNext()
        needNew = False

    for i in range(len(gameCurrent)):
    
        #print("\n-----------------\n")
        
        #print(i)
    
        #print("Board")
        #print(board[0][0])
        board = board
        prevBoard = np.array([[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]])
        np.copyto(prevBoard,board[0][0])
        #board[0][0][0] = 17
        #print(board)
        #print("Copied array")
        #print(prevBoard)
        #print("OLD BOARD")
        #print(len(board[0][0]))
        #gameBoards = np.concatenate((gameBoards,board),axis=0)
        #board[0][3] = [0]
        gameCurrent[i][0] = int(gameCurrent[i][0])
        gameCurrent[i][2] = int(gameCurrent[i][2])
        if i % 2 == 0:
            board[0][1] = [gameCurrent[i][0]-1]
        else:
            board[0][1] = [32-gameCurrent[i][0]]
             
        #print(board[0])
        #print(gameCurrent[i])
        #print(gameCurrent[i][0]-1)
        #print(gameCurrent[i][2]-1)
        if gameCurrent[i] == [1,"-",0] or gameCurrent[i] == [0,"-",1] or gameCurrent[i] == [2,"-",1]:
            needNew = True
            print("Game added")
            break
        
        if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
            board[0][3] = [1]
            #print("LAST")

        #checks for a step notated '-'
        if gameCurrent[i][0] - gameCurrent[i][2] == 3 or gameCurrent[i][0] - gameCurrent[i][2] == -3 or gameCurrent[i][0] - gameCurrent[i][2] == 5 or gameCurrent[i][0] - gameCurrent[i][2] == -5 or gameCurrent[i][0] - gameCurrent[i][2] == 4 or  gameCurrent[i][0] - gameCurrent[i][2] == -4:
            if i % 2 == 0:
                #print(board[0][0])
                piece = board[0][0][gameCurrent[i][0]-1]
                if piece != -1 and piece != -2:
                    input("ERROR 1")
                    quit()
                if gameCurrent[i][2] > 28 :
                    #print("moving")
                    board[0][0][gameCurrent[i][0]-1] = 0
                    board[0][0][gameCurrent[i][2]-1] = -2
                    #print(board[0][0])
                else:
                    #print("moving")
                    board[0][0][gameCurrent[i][0]-1] = 0
                    board[0][0][gameCurrent[i][2]-1] = piece
                
                #if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
                    #board[0][3] = [1]
                    #print("LAST")
                
                if gameCurrent[i][0] - gameCurrent[i][2] == -3:
                    board[0][2] = [0,1,0,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == -4:
                    if gameCurrent[i][0] in [1,2,3,4,9,10,11,12,17,18,19,20,25,26,27,28]:
                        board[0][2] = [0,1,0,0]
                    if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24]:
                        board[0][2] = [1,0,0,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == -5:
                    board[0][2] = [1,0,0,0]
                
                if  gameCurrent[i][0] - gameCurrent[i][2] == 3:
                    board[0][2] = [0,0,1,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == 4:
                    if gameCurrent[i][0] in [9,10,11,12,17,18,19,20,25,26,27,28]:
                        board[0][2] = [0,0,0,1]
                    if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24,29,30,31,32]:
                        board[0][2] = [0,0,1,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == 5:
                    board[0][2] = [0,0,0,1]
                    
                #gameBoards = np.concatenate((gameBoards,board),axis=0)
                
                #(gameBoards[0])

            else:
                piece = board[0][0][gameCurrent[i][0]-1]
                if piece != 1 and piece != 2:
                    input("ERROR 2")
                    quit()
                if gameCurrent[i][2] < 5:
                    #print("moving")
                    board[0][0][gameCurrent[i][0]-1] = 0
                    board[0][0][gameCurrent[i][2]-1] = 2
                    #print(board[0][0])
                else:
    #                print("moving")
                    board[0][0][gameCurrent[i][0]-1] = 0
                    board[0][0][gameCurrent[i][2]-1] = piece

                #if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
                    #board[0][3] = [1]
                    #print("LAST")
                
                if  gameCurrent[i][0] - gameCurrent[i][2] == -3:
                    board[0][2] = [0,0,1,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == -4:
                    if gameCurrent[i][0] in [1,2,3,4,9,10,11,12,17,18,19,20,25,26,27,28]:
                        board[0][2] = [0,0,1,0]
                    if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24]:
                        board[0][2] = [0,0,0,1]
                if gameCurrent[i][0] - gameCurrent[i][2] == -5:
                    board[0][2] = [0,0,0,1]
                
                if  gameCurrent[i][0] - gameCurrent[i][2] == 3:
                    board[0][2] = [0,1,0,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == 4:
                    if gameCurrent[i][0] in [9,10,11,12,17,18,19,20,25,26,27,28]:
                        board[0][2] = [1,0,0,0]
                    if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24,29,30,31,32]:
                        board[0][2] = [0,1,0,0]
                if gameCurrent[i][0] - gameCurrent[i][2] == 5:
                    board[0][2] = [1,0,0,0]
                    
                #gameBoards = np.concatenate((gameBoards,board),axis=0)
                #print(gameBoards[0])
            

        #checks for jump notated 'x'       
        else:
            #checks position change for one jump, it will change 7 or 9 places
            if gameCurrent[i][0] - gameCurrent[i][2] == 7 or gameCurrent[i][0] - gameCurrent[i][2] ==  -7 or gameCurrent[i][0] - gameCurrent[i][2] == 9 or gameCurrent[i][0] - gameCurrent[i][2] == -9:
                if i % 2 == 0:
                    piece = board[0][0][gameCurrent[i][0]-1]
                    if piece == -1:
                        jumpPos = jump[gameCurrent[i][0]-1]
                        for x in range(len(jumpPos)):
                            if jumpPos[x][1] == gameCurrent[i][2]-1:
                                board[0][0][jumpPos[x][0]] = 0
                                board[0][0][gameCurrent[i][0]-1] = 0
                                board[0][0][gameCurrent[i][2]-1] = piece
                                if gameCurrent[i][2] > 28:
                                    board[0][0][gameCurrent[i][2]-1] = -2

                    else:
                        jumpPos = jumpKing[gameCurrent[i][0]-1]
                        for x in range(len(jumpPos)):
                            if jumpPos[x][1] == gameCurrent[i][2]-1:
                                board[0][0][jumpPos[x][0]] = 0
                                board[0][0][gameCurrent[i][0]-1] = 0
                                board[0][0][gameCurrent[i][2]-1] = piece
                    #if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
                        #board[0][3] = [1]
                        #print("LAST")
                        
                    if  gameCurrent[i][0] - gameCurrent[i][2] == -7:
                        board[0][2] = [0,1,0,0]
                    
                    if gameCurrent[i][0] - gameCurrent[i][2] == -9:
                        board[0][2] = [1,0,0,0]
                    
                    if  gameCurrent[i][0] - gameCurrent[i][2] == 7:
                        board[0][2] = [0,0,1,0]
                    
                    if gameCurrent[i][0] - gameCurrent[i][2] == 9:
                        board[0][2] = [0,0,0,1]
                    
                    
                    
                    #gameBoards = np.concatenate((gameBoards,board),axis=0)   
                    
                    #print(gameBoards[0])

                else:
                    #print(gameCurrent[i][0])
                    #print(gameCurrent[i][2])
                    gameCurrent[i][0] = 32 - gameCurrent[i][0]
                    #print(gameCurrent[i][0])
                    gameCurrent[i][2] = 32 - gameCurrent[i][2]
                    #print(gameCurrent[i][2])
                    
                    tempBoard = np.array([flipBoard(board[0])])
                    #print("BOARDS:")
                    #print(board[0][0])
                    #print(tempBoard[0])
                    
                    piece = tempBoard[0][gameCurrent[i][0]]
                    
                    if piece == -1:
                        jumpPos = jump[gameCurrent[i][0]]
                        for x in range(len(jumpPos)):
                            if jumpPos[x][1] == gameCurrent[i][2]:
                                #print(tempBoard[0][gameCurrent[i][0]])
                                #print(gameCurrent[i][0])
                                tempBoard[0][jumpPos[x][0]] = 0
                                tempBoard[0][gameCurrent[i][0]] = 0
                                tempBoard[0][gameCurrent[i][2]] = piece
                                if gameCurrent[i][2] > 27:
                                    tempBoard[0][gameCurrent[i][2]] = -2
                        board[0][0] = flipBoard(tempBoard)
                        #print("Board after jump")
                        #print(board)

                    else:
                        jumpPos = jumpKing[gameCurrent[i][0]]
                        for x in range(len(jumpPos)):
                            if jumpPos[x][1] == gameCurrent[i][2]:
                                tempBoard[0][jumpPos[x][0]] = 0
                                tempBoard[0][gameCurrent[i][0]] = 0
                                tempBoard[0][gameCurrent[i][2]] = piece
                                #print("Done king jump")
                    
                        board[0][0] = flipBoard(tempBoard)
                    
                    #if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
                       # board[0][3] = [1]
                        #print("LAST")
                        
                    if  gameCurrent[i][0] - gameCurrent[i][2] == -7:
                        board[0][2] = [0,0,1,0]
                    
                    if gameCurrent[i][0] - gameCurrent[i][2] == -9:
                        board[0][2] = [0,0,0,1]
                    
                    if  gameCurrent[i][0] - gameCurrent[i][2] == 7:
                        board[0][2] = [0,1,0,0]
                    
                    if gameCurrent[i][0] - gameCurrent[i][2] == 9:
                        board[0][2] = [1,0,0,0]
                    
                    
                    #print(gameBoards[0])

### Might not work yet
            else:
                if i % 2 == 0:
                    #because it only shows start and finish, checking has to be done to see if it jumps more than once
                    piece = board[0][0][gameCurrent[i][0]-1]
                    
                    if piece == -1:
                        player = False
                        possibleMoves = checkMove(gameCurrent[i][0]-1,gameCurrent[i][2]-1,i,gameCurrent,board[0],player)
                        #print("doing something")
                        if len(possibleMoves) == 0:
                            print("FAILED")
                        #print("PossibleMoves:",possibleMoves)
                        piece = board[0][0][gameCurrent[i][0]-1]
                        board[0][0][gameCurrent[i][0]-1] = 0
                        for y in range(len(possibleMoves)-1):
                            board[0][0][possibleMoves[y]] = 0
                        board[0][0][gameCurrent[i][2]-1] = piece
                        if gameCurrent[i][2] > 28:
                            board[0][0][gameCurrent[i][2]-1] = -2
                        
                    if piece == -2:
                        player = False
                        #print("king")
                        possibleMoves = checkMoveKing(gameCurrent[i][0]-1,gameCurrent[i][2]-1,i,gameCurrent,board[0],player)
                        if len(possibleMoves) == 0:
                            print("FAILED")
                        piece = board[0][0][gameCurrent[i][0]-1]
                        board[0][0][gameCurrent[i][0]-1] = 0
                        for y in range(len(possibleMoves)-1):
                            board[0][0][possibleMoves[y]] = 0
                        board[0][0][gameCurrent[i][2]-1] = piece
                             
                    #print(possibleMoves)                
                    if  gameCurrent[i][0] - possibleMoves[0]+1 == -3:
                        board[0][2] = [0,1,0,0]
                    if gameCurrent[i][0] - possibleMoves[0]+1 == -4:
                        if gameCurrent[i][0] in [1,2,3,4,9,10,11,12,17,18,19,20,25,26,27,28]:
                            board[0][2] = [0,1,0,0]
                        if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24]:
                            board[0][2] = [1,0,0,0]
                    if gameCurrent[i][0] - possibleMoves[0]+1 == -5:
                        board[0][2] = [1,0,0,0]
                    
                    if  gameCurrent[i][0] - possibleMoves[0]+1 == 3:
                        board[0][2] = [0,0,1,0]
                    if gameCurrent[i][0] - possibleMoves[0]+1 == 4:
                        if gameCurrent[i][0] in [9,10,11,12,17,18,19,20,25,26,27,28]:
                            board[0][2] = [0,0,0,1]
                        if gameCurrent[i][0] in [5,6,7,8,13,14,15,16,21,22,23,24,29,30,31,32]:
                            board[0][2] = [0,0,1,0]
                    if gameCurrent[i][0] - possibleMoves[0]+1 == 5:
                        board[0][2] = [0,0,0,1]
                    
                        
                else:
                    #print(gameCurrent[i][0])
                    #print(gameCurrent[i][2])
                    gameCurrent[i][0] = 32 - gameCurrent[i][0]
                    #print(gameCurrent[i][0])
                    gameCurrent[i][2] = 32 - gameCurrent[i][2]
                    #print(gameCurrent[i][2])
                    
                    tempBoard = np.array([flipBoard(board[0])])
                    #print("BOARDS:")
                    #print(board[0])
                    #print(tempBoard[0])
                    
                    piece = tempBoard[0][gameCurrent[i][0]]

                    #print("Player 2 ")
                    #print(piece)
                    player = True
                    
                    if piece == -1:
                        possibleMoves = checkMove(gameCurrent[i][0],gameCurrent[i][2],i,gameCurrent,tempBoard,player)
                        #print("doing something")
                        if len(possibleMoves) == 0:
                            print("FAILED")
                            
                        #print("PossibleMoves:",possibleMoves)
                        piece = tempBoard[0][gameCurrent[i][0]]
                        tempBoard[0][gameCurrent[i][0]] = 0
                        for y in range(len(possibleMoves)-1):
                            tempBoard[0][possibleMoves[y]] = 0
                        tempBoard[0][gameCurrent[i][2]] = piece
                        if gameCurrent[i][2] > 27:
                            tempBoard[0][gameCurrent[i][2]] = -2
                        board[0][0] = flipBoard(tempBoard)
                        
                        
                    if piece == -2:
                        #print("king")
                        possibleMoves = checkMoveKing(gameCurrent[i][0],gameCurrent[i][2],i,gameCurrent,tempBoard,player)
                        #print("doing something")
                        #print("PossibleMoves:",possibleMoves)
                        piece = tempBoard[0][gameCurrent[i][0]]
                        tempBoard[0][gameCurrent[i][0]] = 0
                        for y in range(len(possibleMoves)-1):
                            tempBoard[0][possibleMoves[y]] = 0
                        tempBoard[0][gameCurrent[i][2]] = piece
                        board[0][0] = flipBoard(tempBoard)
                        
                    if  gameCurrent[i][0] - possibleMoves[0] == -3:
                        board[0][2] = [0,1,0,0]
                    if gameCurrent[i][0] - possibleMoves[0] == -4:
                        if gameCurrent[i][0] in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
                            board[0][2] = [0,1,0,0]
                        if gameCurrent[i][0] in [4,5,6,7,12,13,14,15,20,21,22,23]:
                            board[0][2] = [1,0,0,0]
                    if gameCurrent[i][0] - possibleMoves[0] == -5:
                        board[0][2] = [1,0,0,0]
                    
                    if  gameCurrent[i][0] - possibleMoves[0] == 3:
                        board[0][2] = [0,0,1,0]
                    if gameCurrent[i][0] - possibleMoves[0] == 4:
                        if gameCurrent[i][0] in [8,9,10,11,16,17,18,19,24,25,26,27]:
                            board[0][2] = [0,0,0,1]
                        if gameCurrent[i][0] in [4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31]:
                            board[0][2] = [0,0,1,0]
                    if gameCurrent[i][0] - possibleMoves[0] == 5:
                        board[0][2] = [0,0,0,1]
                        
                    
                        
            #if gameCurrent[i+1] == [1,"-",0] or gameCurrent[i+1] == [0,"-",1] or gameCurrent[i+1] == [2,"-",1]:
                #board[0][3] = [1]
                #print("LAST")
            
        #print("Old board1")
        #print(prevBoard)                    
        addBoard = np.array([[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],[0],[0,0,0,0],[1]]])
        np.copyto(addBoard,board)
        
        #addBoard[0][0] = prevBoard[0][0]
        if i % 2 == 0:
            addBoard[0][0] = prevBoard[0]
        else:
            #print("flipped",flipBoardFinal(prevBoard[0]))
            addBoard[0][0] = flipBoardFinal(prevBoard[0])
        #print("Old board2")
        #print(prevBoard)
        #print("Board before")
        #print(addBoard)
        #print("Board after")
        #print(board[0][0])
        gameBoards = np.concatenate((gameBoards,addBoard),axis=0)
    
                
                                  
    return gameBoards    

def checkMove(start,end,currentMove,gameCurrent,board,player):
    #print(board[0])
    jump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
 
    possibleMoves = []
    #print("start", start)
    #print("end",end)
    

    if start not in jump.keys():
            return possibleMoves
    #thisJump = jump[start]
    #print(thisJump)
         
    for i in range(len(jump[start])):
        moves = []


        #print("jumped:", thisJump[i][0])
    
        
        if board[0][jump[start][i][0]] > 0:
            moves.append(jump[start][i][0])
            #print("Passed enemy pos")
       
        else:
            #print("Failed enemy pos")
            continue
    
        #print("Jump info")
        #print("counter:",board[0][jump[start][i][1]])
        #print("pos:",jump[start][i][1])
        #print(jump[start][i][1])
        if board[0][jump[start][i][1]] == 0:
            moves.append(jump[start][i][1])
            #print("Passed jump pos")
        else:
            #print("Failed jump pos")
            continue
            
        
        if counterMoves(currentMove,jump[start][i][0],gameCurrent,player) == True:
            continue
        #print("!")
        
        if jump[start][i][1] != end and pos(jump[start][i][1]) == -1:
            print("checking for -1")
            continue
        
        if jump[start][i][1] == end:
            possibleMoves = (possibleMoves+moves)
            #print("done")
            return possibleMoves
            #print("done")
                
        else:
            deeperMoves = checkMove(jump[start][i][1],end,currentMove,gameCurrent,board,player)
            if len(deeperMoves) > 0:
                moves = (moves + deeperMoves)
                possibleMoves = (possibleMoves + moves)
                #print("done deeper")
                return possibleMoves
                
                
    if len(possibleMoves) > 1:
        print("Problem")
        
    return possibleMoves    
                
def checkMoveKing(start,end,currentMove,gameCurrent,board,player):
    #print(board[0])
    jump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
 
    possibleMoves = []
    #print("start", start)
    #print("end",end)
    
    if start not in jump.keys():
        return possibleMoves
    thisJump = jump[start]
    #print(thisJump)
         
    for i in range(len(thisJump)):
        moves = []
        
        #print("jumped:", thisJump[i][0])
    
        
        if board[0][jump[start][i][0]] > 0:
            moves.append(jump[start][i][0])
            #print("Passed enemy pos")
       
        else:
            #print("enemy pos failed")
            continue
    
        #print(board[0][jump[start][i][1]])
        #print(jump[start][i][1])
        
        if board[0][jump[start][i][1]] == 0:
            moves.append(jump[start][i][1])
            #print("Passed jump pos")
        else:
            #print("failed jump pos")
            continue
        
        
        if counterMoves(currentMove,jump[start][i][0],gameCurrent,player) == True:
            continue
        
        #print("!")
        
        #if jump[start][i][1] != end and pos(jump[start][i][1]) == -1:
            #print("checking for -1")
            #continue
        
        if jump[start][i][1] == end:
            possibleMoves = (possibleMoves+moves)
            #print("done")
            return possibleMoves
            #print("done")
                
        else:
            deeperMoves = checkMoveKing(jump[start][i][1],end,currentMove,gameCurrent,board,player)
            if len(deeperMoves) > 0:
                moves = (moves + deeperMoves)
                possibleMoves = (possibleMoves + moves)
                #print("done deeper")
                return possibleMoves
                
                
    if len(possibleMoves) > 1:
        print("Problem")
        
    return possibleMoves    
kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [21,17], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [23,18], 28 : [24,21], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [26,22]}
            

def flipBoard(board):
    #print("FLIPPING")
    tempBoard = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    #print("board")
    #print(board)
    for i in range(32):
        tempBoard[i] = -(board[0][31-i])
    
    return tempBoard
def flipBoardFinal(board):
    #print("FLIPPING END")
    tempBoard = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    #print("board")
    #print(board)
    for i in range(32):
        tempBoard[i] = -(board[31-i])
        
    #print(tempBoard)
    
    return tempBoard
            

gameBoards = np.array([[[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],[0],[0,0,0,0],[1]]])
              
#9377    
iList = []
for i in range(9375):
    iList.append(i)
    print("Game:", i)    
    #getReadGame(True)
    try:
        gameBoards = np.concatenate((gameBoards,getReadGame(True)),axis=0)
    
    except:
        print(len(iList))
file = open("moveFile.txt","wb")
pickle.dump(gameBoards,file)
file.close()
   
