# coding=utf-8
import numpy as np


class labyrinth(object):
    def __init__(self):
        self.state = 11
        self.reward = np.ones((12,))*(-0.02)
        self.reward[3] = 1
        self.reward[7] = -1
        self.step = np.array(range(12)*4).reshape(4,12).transpose()
        for i in range(12):
            self.step[i][0] = self.step[i][0] - 4
            self.step[i][1] = self.step[i][1] + 1
            self.step[i][2] = self.step[i][2] + 4
            self.step[i][3] = self.step[i][3] - 1
        for i in range(12):
            for j in range(4):
                if self.step[i][j] < 0 or self.step[i][j] > 11 or self.step[i][j] == 5:
                    self.step[i][j] = i
        for i in range(12):
            if i % 4 == 0:
                self.step[i][3] = i
            if i % 4 == 3:
                self.step[i][1] = i
        
    def move(self, action):
        if self.state == 3 or self.state == 7:
            return -1, -1, True
        rand = np.random.random()
        index = action.argmax()
        if rand <= 0.8:
            self.state = self.step[self.state][index]
        elif rand <= 0.9:
            self.state = self.step[self.state][index-1]
        else:
            self.state = self.step[self.state][(index+1)%4]
        state = np.zeros((11,))
        if self.state > 4:
            state[self.state-1] = 1
        else:
            state[self.state] = 1
        if self.state == 3 or self.state == 7:
            terminal = True
        else:
            terminal = False
            
        return state, self.reward[self.state], terminal
    
    def reset(self):
        self.state = 11
    
    
if __name__ == '__main__':
    laby = labyrinth()
    for i in range(30):
        laby.reset()
        while 1:
            print laby.state,
            rand = np.random.randint(4)
            step = np.zeros((4,))
            step[rand] = 1
            result = laby.move(step)
            if result == -1:
                break
        print 