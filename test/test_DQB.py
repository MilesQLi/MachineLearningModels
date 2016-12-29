import sys
sys.path.append('..')

from with_tensorflow.BrainDQN_NIPS import BrainDQN
import numpy as np
from datasets.labyrinth_for_reinforcement import *

def playLabyrinth():
    # Step 1: init BrainDQN
    actions = 4
    brain = BrainDQN(actions,[11, 4])
    # Step 2: init Flappy Bird Game
    laby = labyrinth()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([0]*4)  # do nothing
    randi = np.random.randint(4)
    action0[randi] = 1
    observation0, reward0, terminal = laby.move(action0)
    brain.setInitState(observation0)
    i = 0
    #Step 3.2: run the game
    states = []
    while i < 10000:
        print 'epoch:',i
        i += 1
        states.append(laby.state)
        action = brain.getAction()
        nextObservation,reward,terminal = laby.move(action)
        #print 'trained',i,'episodes','nextObservation:',nextObservation,'\r',
        brain.setPerception(nextObservation,action,reward,terminal,states)
        if terminal == True:
            states.append(laby.state)
            print states
            laby.reset()
            states = []
            continue
    for i in range(11):
        tmp = np.array([0]*11)
        tmp[i] = 1
        print '------------'
        print i
        print brain.getQValueFromState(tmp)
    for i in range(11):
        tmp = np.array([0]*11)
        tmp[i] = 1
        print '------------'
        print i
        print brain.getActionFromState(tmp)
    #===========================================================================
    # print '------------'
    # print 'w',len(brain.Ws)
    # print brain.session.run(brain.Ws[0])
    # print '------------'
    # print 'b',len(brain.bs)
    # print brain.session.run(brain.bs[0])
    #===========================================================================
      
if __name__ == '__main__':
    playLabyrinth()