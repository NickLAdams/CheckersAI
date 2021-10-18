import tkinter as tk
import numpy as np
from keras.models import load_model


class GameBoard(tk.Frame):
    def __init__(self, parent, rows=8, columns=8, size=32, colour1="white", colour2="black"):
        

        self.rows = rows
        self.columns = columns
        self.size = size
        self.colour1 = colour1
        self.colour2 = colour2
        #These are the coordinates of the counters
        self.places = {0 : (60,10,90,40), 1 : (160,10,190,40), 2 : (260,10,290,40), 3 : (360,10,390,40), 
                       4 : (10,60,40,90), 5 : (110,60,140,90), 6 : (210,60,240,90) , 7 : (310,60,340,90) , 
                       8 : (60,110,90,140), 9 : (160,110,190,140), 10 : (260,110,290,140) , 11 : (360,110,390,140), 
                       12 : (10,160,40,190), 13 : (110,160,140,190), 14 : (210,160,240,190), 15 : (310,160,340,190), 
                       16 : (60,210,90,240), 17 : (160,210,190,240), 18 : (260,210,290,240), 19 : (360,210,390,240),
                       24 : (60,310,90,340), 25 : (160,310,190,340), 26 : (260,310,290,340), 27 : (360,310,390,340), 
                       28 : (10,360,40,390), 29 : (110,360,140,390), 30 : (210,360,240,390), 31 : (310,360,340,390),
                       20 : (10,260,40,290), 21 : (110,260,140,290), 22 : (210,260,240,290), 23 : (310,260,340,290)}
        
        
        
        canvas_width = columns * size
        canvas_height = rows * size

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                width=canvas_width, height=canvas_height)
        
        self.canvas.pack(side="top", fill="both",expand=1, padx=2, pady=2)
        
    

        

        
        #This build the board background with the black and white squares
        self.canvas.bind("<Configure>", self.makeBoard)
	
    #This function takes the input of the user and makes their move, it will then pass onto the AiMove and make the AI take a move
    def movePiece(self):
        validList = checkers.validMoves()
        #checks the make sure the inputs are valid
        try:
            start = int(e1.get())
            valid = True
        except:
            top = tk.Toplevel(root)
            tk.Label(top, text="Please enter a counter").pack()
            ok = tk.Button(top, text="OK", command=top.destroy)
            ok.pack(pady=5)
            valid = False
            
        #end = int(e2.get())
        if valid:
            
            index = myList.curselection()
            if index == ():
                top = tk.Toplevel(root)
                tk.Label(top, text="Please enter a direction").pack()
                ok = tk.Button(top, text="OK", command=top.destroy)
                ok.pack(pady=5)
                
            else:
                
                direction = np.zeros(4)
                
                if start > 31 or start < 0:
                    top = tk.Toplevel(root)
                    tk.Label(top, text="Not a valid counter").pack()
                    ok = tk.Button(top, text="OK", command=top.destroy)
                    ok.pack(pady=5)
                    
                else:
                
                    if validList[start][index] == 0:
                        top = tk.Toplevel(root)
                        tk.Label(top, text="Not a valid move").pack()
                        ok = tk.Button(top, text="OK", command=top.destroy)
                        ok.pack(pady=5)
                        
                        
                    if validList[start][index] == 1:    
                        direction[index] = 1
                        counter = start
                        #end = start + jump
                        #self.canvas.coords(self.pos[start], self.places[end])
                        jumpAnother , validList = checkers.makeMove(counter,direction)
                        board32 = checkers.getBoard32()
                        self.updateBoard(board32)
                        if jumpAnother == True:
                            top = tk.Toplevel(root)
                            tk.Label(top, text="You have another move").pack()
                            ok = tk.Button(top, text="OK", command=top.destroy)
                            ok.pack(pady=5)
                            
                        else:
                            result = checkers.checkWin()
                            if result == 1:
                                self.endGame(True)
                            if result == -1:
                                self.endGame(False)
                            if result == 0:
                                checkers.flipBoard()
                                board = checkers.getBoard64()
                                #passes onto the aiMove after 1.5 seconds 
                                root.after(1500,self.aiMove,board)
    
    #makes the AI move, it sends the board to the network and makes it predict the best move for that situation. 
    def aiMove(self,board):
        loss = False
        valid_places = checkers.validMoves()
        board = np.reshape(board,(1,8,8,1))
        q = model.predict(board)
        q = q.reshape((32,4))
        valid_places_index = np.nonzero(valid_places)# look at valid positions
        if valid_places_index[0].size == 0:
            loss = True
        else:    
            max_Q_valid = np.unravel_index(np.argmax(q[valid_places_index]),q[valid_places_index].shape) # find max Q only in valid positions - index is for sublist
            action_index = [valid_places_index[0][max_Q_valid],valid_places_index[1][max_Q_valid]]
        if loss == False:
            print("AI move",action_index)
            counter = action_index[0]
            direction = int(action_index[1])
            checkers.makeMoveAI(counter,direction,False)
            result = checkers.checkWin()
            if result == 1:
                print("player lost")
                self.endGame(False)
            
            if result == -1:
                print("player won")
                self.endGame(True)
            
        else:
            self.endGame(True)
        checkers.flipBoard()
        board = checkers.getBoard32()
        self.updateBoard(board)
     
    #THis is the function that gets called if someone wins the game, creates a pop up window telling the user who won
    def endGame(self,winner):
        if winner == True:
            self.top = tk.Toplevel(root)
            tk.Label(self.top, text="Well done! You beat the AI!").pack()
            ok = tk.Button(self.top, text="Play Again", command=self.finish)
            ok.pack(pady=5)
            
        if winner == False:
            self.top = tk.Toplevel(root)
            tk.Label(self.top, text="Unlucky! You Lost!").pack()
            ok = tk.Button(self.top, text="Play Again", command=self.finish)
            ok.pack(pady=5)
     
    #This is the function that runs when the user presses 'play again' on the pop up window when the game is over, it'll reset the board and remove pop up
    def finish(self):
        self.top.destroy()
        self.resetBoard()
            
    #Updates the board, it deletes all the canvases with the tag 'counter' and then recreates them in the new postions, it uses the dictionary from before to get the coordinates for them
    
    def updateBoard(self,board):
        print("updating board")
        self.canvas.delete("counter")
        for i in range(32):
            if board[i] == -1:
                self.canvas.create_oval(self.places[i],fill="deep sky blue",tag="counter")
            if board[i] == -2:
                self.canvas.create_oval(self.places[i],fill="blue2",tag="counter")
                
                    
            if board[i] == 1:
                self.canvas.create_oval(self.places[i],fill="yellow",tag="counter")
            if board[i] == 2:
                self.canvas.create_oval(self.places[i],fill="dark orange",tag="counter")
                
            self.canvas.create_text((self.places[i][0]+15,self.places[i][1]+15),text=i)
        print("done")
        
    #This resets the board both with the checkers class and on the GUI, making sure everthing is in sync    
    def resetBoard(self):
        print("Reset")
        checkers.reset()
        board = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        self.canvas.delete("counter")
        for i in range(32):
            if board[i] == -1:
                self.canvas.create_oval(self.places[i],fill="deep sky blue",tag="counter")
            if board[i] == -2:
                self.canvas.create_oval(self.places[i],fill="blue2",tag="counter")
                
                    
            if board[i] == 1:
                self.canvas.create_oval(self.places[i],fill="yellow",tag="counter")
            if board[i] == 2:
                self.canvas.create_oval(self.places[i],fill="dark orange",tag="counter")
            self.canvas.create_text((self.places[i][0]+15,self.places[i][1]+15),text=i)
        
	#This creates the board background with alternating colour squares
    def makeBoard(self, event):
        xsize = int((400) / self.columns)
        ysize = int((400) / self.rows)
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        colour = self.colour2
        for row in range(self.rows):
            colour = self.colour1 if colour== self.colour2 else self.colour2
            for col in range(self.columns):
                x1 = (col * self.size)
                y1 = (row * self.size)
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", fill=colour, tags="square")
                colour = self.colour1 if colour == self.colour2 else self.colour2
        self.canvas.tag_lower("square")
        
    #This creates a pop up window when the help button is clicked and gives information about the game    
    def help(self):
        top = tk.Toplevel(root)
        tk.Label(top, text="""
How to play:
-Each player has 12 counters and they take it in turn to move one piece.
-A move consists of either a step or a jump.
-A step is just a move into an adjacent square.
-A jump is when you take the opponents piece in the move.
-You must take a jump if it is available.
-Once you get a counter to the last row it will then become a king.
-A king can move both forwards and backwards.
-The winner is the player to take all but 2 of the opponents pieces.

How to move:
-You play as the blue counters and are moving down the board.
-Each counter is labelled with their position.
-Enter the position and the direction you want to move.
-The AI will move after you make your move.
-Kings are indicated by a darker shade of the colour.""").pack()
        ok = tk.Button(top, text="OK", command=top.destroy)
        ok.pack(pady=5)
        
    #master = tk.Frame(self)
    #master.pack()
    #b = tk.Button(self, text="Reset Board", command = setpiece(self))
    #self.b.pack()
    
class checkers:
    #This is the class that deals with playing the checkers game
    
    #creates starting config of the boards
    def __init__(self):
        self.board64 = np.array([[0,-1,0,-1,0,-1,0,-1],[-1,0,-1,0,-1,0,-1,0],[0,-1,0,-1,0,-1,0,-1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]])
        self.board32 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        #self.board32 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,-2])
        #self.update64()
     
    #Resets the board configs
    def reset(self):
        self.board64 = np.array([[0,-1,0,-1,0,-1,0,-1],[-1,0,-1,0,-1,0,-1,0],[0,-1,0,-1,0,-1,0,-1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1],[1,0,1,0,1,0,1,0]])
        self.board32 = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        
    #This will find the amount that will need to be added to the counter based on what direction it is going to find the square it will land in. THat will then be used to check what counter is in the square and what move will be made    
    def findValues(self, counter, direction):
        self.counter = counter 
        if type(direction) == int:
            self.direction = direction
        else:
            self.direction = np.argmax(direction)
        
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
    
    #flips the board so the player taking the move is always going down, this is just so it is fed into the network in the correct orientation
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
    
    #This finds the valid moves available for the current player for this go. Creates a 23x4 array with a 1 in each valid move, the 32 dimension represents the counter postion and the 4 dimension represents the direction the counter is moving in. This is just done by searching the dictionaries for the counter and move postion
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
        kingStepDir = {4 : [2], 5:[3,2],6:[3,2],7:[3,2,0],0: [1], 8: [3,1] , 15: [2,0], 16 : [3,1], 23 : [2,1],24 :[1,0], 25: [1,0],26:[1,0],27:[1], 31: [3,2]}
        jumps = 0
        for i in range(32):
            if self.board32[i] == -1 and i < 24:
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
    
    #This makes the move for the ai, with the only difference from the user move is that the AI will have to run again if it has another jump available but the user just makes the move again. 
    def makeMoveAI(self,counter,direction,jump):
        if jump == False:
            self.points = 0
        counter , jump = self.findValues(counter,direction)
        #print(counter,jump)
        kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
        normalJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
        
        anotherJump = False
        
        endPos = counter + jump
        #print(counter, jump, endPos)
        piece = self.board32[counter]
        if self.board32[endPos] == 0:
            self.board32[counter] = 0
            if endPos > 27:
                piece = -2
                self.points += 0.5
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
                            self.points += 0.5
                        self.board32[normalJump[counter][i][1]] = piece
                        self.points += 0.5
                        #print("2")           
                        anotherJump , self.validList = self.checkJump(normalJump[counter][i][1])
            else:
                for i in range(len(kingJump[counter])):
                    if kingJump[counter][i][0] == endPos:
                        self.board32[counter] = 0
                        self.board32[endPos] = 0
                        self.board32[kingJump[counter][i][1]] = piece
                        self.points += 0.5
                        #print("3")
                        anotherJump , self.validList = self.checkJump(kingJump[counter][i][1])
        self.update64()
        if anotherJump == True:
            nextMove = np.unravel_index(np.argmax(self.validList),self.validList.shape)
            self.makeMoveAI(int(nextMove[0]),int(nextMove[1]),True)
    
    #This carries out the move for the user, updates the board and returns if it has another jump or not. 
    def makeMove(self,counter,direction):
        counter , jump = self.findValues(counter,direction)
        kingJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17],[5,1]], 9 : [[5,0],[6,2],[13,16],[14,18]], 10 : [[6,1],[7,3],[14,17],[15,19]], 11 : [[7,2],[15,18]], 12 : [[8,5],[16,21]], 13 : [[8,4],[9,6],[16,20],[17,22]], 14 : [[9,5],[10,7],[17,21],[18,23]], 15 : [[10,6],[18,22]], 16 : [[13,9],[21,25]], 17 : [[13,8],[14,10],[21,24],[22,26]], 18 : [[14,9],[15,11],[22,25],[23,27]], 19 : [[15,10],[23,26]], 20 : [[16,13],[24,29]], 21 : [[16,12],[17,14],[24,28],[25,30]], 22 : [[17,13],[18,15],[25,29],[26,31]], 23 : [[18,14],[26,30]], 24 : [[21,17]], 25 : [[21,16],[22,18]], 26 : [[22,17],[23,19]],27 : [[23,18]], 28 : [[24,21]], 29 : [[24,20],[25,22]], 30 : [[25,21],[26,23]], 31 : [[26,22]]}
        normalJump = {0 : [[5,9]], 1 : [[5,8],[6,10]], 2 : [[6,9],[7,11]], 3 : [[7,10]], 4 : [[8,13]], 5 : [[8,12],[9,14]], 6 : [[9,13],[10,15]], 7 : [[10,14]], 8 :[[13,17]], 9 : [[13,16],[14,18]], 10 : [[14,17],[15,19]], 11 : [[15,18]], 12 : [[16,21]], 13 : [[16,20],[17,22]], 14 : [[17,21],[18,23]], 15 : [[18,22]], 16 : [[21,25]], 17 : [[21,24],[22,26]], 18 : [[22,25],[23,27]], 19 : [[23,26]], 20 : [[24,29]], 21 : [[24,28],[25,30]], 22 : [[25,29],[26,31]], 23 : [[26,30]]}
        
        anotherJump = False
        endPos = counter + jump
        piece = self.board32[counter]
        if self.board32[endPos] == 0:
            self.board32[counter] = 0
            if endPos > 27:
                piece = -2
            self.board32[endPos] = piece
        else:
            if piece == -1:
                for i in range(len(normalJump[counter])):
                    if normalJump[counter][i][0] == endPos:
                        self.board32[counter] = 0
                        self.board32[endPos] = 0
                        if normalJump[counter][i][1] > 27:
                            piece = -2
                        self.board32[normalJump[counter][i][1]] = piece
                        anotherJump , self.validList = self.checkJump(normalJump[counter][i][1])
            else:
                for i in range(len(kingJump[counter])):
                    if kingJump[counter][i][0] == endPos:
                        self.board32[counter] = 0
                        self.board32[endPos] = 0
                        self.board32[kingJump[counter][i][1]] = piece
                        anotherJump , self.validList = self.checkJump(kingJump[counter][i][1])
                        
        self.update64()
        return anotherJump, self.validList
    
    #This checks if there is another jump available after the first jump, it's very similar to the validMoves function but it just looks for the valid jumps for the counter that just made the move                    
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
    
    #updates the 8x8 board to keep it the same as the 32x1 board
    def update64(self):
        temp64 = np.zeros(64)
        for x in range(32):
            if x in [0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27]:
                temp64[2*x+1] = self.board32[x]
            else:
                temp64[2*x] = self.board32[x]
                
        temp64 = np.reshape(temp64,(8,8))
        self.board64 = temp64
      
    #Checks if one of the players has less than 3 counters and then returns the winner  
    def checkWin(self):
        p1 = 0
        p2 = 0
        
        for i in range(32):
            if self.board32[i] < 0:
                p1 += 1
            if self.board32[i] > 0:
                p2 += 1
        #print("p1:",p1,"p2:",p2)
        
        if p1 < 3:
            return -1
        if p2 < 3:
            return 1
        else:
            return 0 
    
    #Returns the 8x8 board
    def getBoard64(self):
        return self.board64
    
    #returns the 32x1 board
    def getBoard32(self):
        return self.board32


if __name__ == "__main__":
  #loads in model and weights for the network
    model = load_model("my_model_1024.h5")
    model.load_weights("weights_1024N.h5")
    #initialise checkers class
    checkers = checkers()
    #creates the gui board and window that is 620x420
    root = tk.Tk()
    root.title("Checkers")
    #adds a frame for all the buttons and inputs to go into
    buttons = tk.Frame(root)
    buttons.pack(side="right")
    board = GameBoard(root)
    root.geometry("620x420")
    root.resizable(width=False,height=False)
    board.pack(side="top", fill="both", expand=1, padx=4, pady=4)
    board.resetBoard() 
    #creates buttons, input and title
    label = tk.Label(buttons, text="""Welcome to my
checkers AI""", fg = "blue", font = ("verdana 17 underline bold"))
    label.pack()
    label = tk.Label(buttons, text="Counter")
    label.pack()
    e1 = tk.Entry(buttons)
    e1.focus_set()
    e1.pack()
    #e2 = tk.Entry(buttons)
    #e2.pack()
    label = tk.Label(buttons, text="Direction")
    label.pack()
    moveList = ["Down-Right","Down-Left","Up-Right","Up-Left"]
    myList = tk.Listbox(buttons,height=4,selectmode = "browse")
    for each in moveList:
        myList.insert(tk.END,each)
    myList.pack()
    
    label = tk.Label(buttons, text=" ")
    label.pack()

    move = tk.Button(buttons, text ="MakeMove",width = 25, command=board.movePiece)
    move.pack()
    
    b = tk.Button(buttons, text="Reset Board",width = 25, command=board.resetBoard)
    b.pack()
    
    label = tk.Label(buttons, text=" ")
    label.pack()
    
    h = tk.Button(buttons, text="-Help-",width = 25, command=board.help)
    h.pack()
    
    root.mainloop()
