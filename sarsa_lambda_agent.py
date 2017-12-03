#!/usr/bin/env python


# Avery Tan altan 1392212
# CMPUT366 A7


"""
  Author: Adam White, Matthew Schlegel, Mohammad M. Ajallooeian, Sina Ghiassian
  Purpose: Skeleton code for Monte Carlo Exploring Starts Control Agent
           for use on A3 of Reinforcement learning course University of Alberta Fall 2017
 
"""

from utils import rand_in_range, rand_un
from tiles3 import *
import numpy as np
import pickle
import operator
import random






curr_state = None
old_state = None
last_action = None
iht=None


mem_size = None
num_tilings = None
tile_width = None
alpha = None
lambd = None
epsilon = None
w=None
gamma = None
z = None







def get_RMSE():
    """
    returns rmse and state value estimates
    """
    global w, true_val, num_tilings, feature_vector,iht, tile_width

    value_list = list()

    total_error = np.zeros(1)
    for i in range(1000):
        v= 0.0 #set value of current state to 0
        indices = tiles(iht,50,[(i+1)/1000.0*1/tile_width])
        for j in indices:
            v+=w[j]
        total_error[0] += ((v-true_val[i+1])**2)
        value_list.append(v)
    total_error/=1000
    rmse = total_error**(0.5)

    # print rmse
    return (rmse[0],value_list)


    



def agent_init():
    """
    Hint: Initializes Q, where we store the Q(s,a). Q is a 2d array such that Q[x][y][a] where x and y are
    intergers and represent a particular location and a is an action and Q[x][y][a] is the state-action value for
    that particular action taken in that particular state
    Returns: nothing
    """
    global w, num_tilings, mem_size, tile_width, alpha, gamma, lambd, epsilon, iht 
    
    num_tilings = 8
    mem_size = 4096
    tile_width = 0.125
    alpha = 0.1/num_tilings
    lambd = 0.9
    epsilon = 0.0
    gamma = 1.0


    w = np.zeros(mem_size)
    for i in range(mem_size):
    	w[i] = random.uniform(-0.001,0)


    iht = IHT(mem_size)


def agent_start(state):
    """
    returns random action
    """
    global curr_state, last_action, old_state, action_list, z

    z = np.zeros(mem_size)

    curr_state = state 
    action = random.randrange(-1,2)
    

    old_state=state
    last_action=action

    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: int between 1 and 1000
    Returns action as an int between -100 and 100
    """
    global  old_state, iht, curr_state, last_action, alpha, w, gamma, num_tilings, tile_width, epsilon
    
    curr_state = state

    #unpack the np arrays
    x,xdot = state
    o_x, o_xdot =  old_state

    action = random.randrange(-1,2)

    
    #calculate indices of affected features of curr_state and old_state
    features_old = tiles(iht,num_tilings,[((o_x/(0.5+1.2))*(1/tile_width)), ((o_xdot/(0.07+0.07))*(1/tile_width))],[last_action]) #x(s) components. have 50 of them
    features_curr = tiles(iht,num_tilings,[((x/(0.5+1.2))*(1/tile_width)), ((xdot/(0.07+0.07))*(1/tile_width))], [action]) #x(s') components. have 50 of them
    gradient_indices = features_old 


    #calculate state value for old_state
    v_old = 0.0
    for ii in features_old:
        v_old+=w[ii]

    #calculate state value for curr_state
    v_curr = 0.0
    for ij in features_curr:
        v_curr+=w[ij]
    
    error = alpha*(reward+gamma*v_curr-v_old)
       

    #update weights
    for ik in gradient_indices:
        w[ik] = w[ik]+error

    old_state = state
    last_action = action
    


    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global w, feature_vector, old_state, curr_state, iht, num_tilings, alpha, gamma


    features_old = tiles(iht,50,[old_state/1000.0*1/tile_width]) #x(s) components. have 50 of them
    features_curr = tiles(iht,50,[curr_state/1000.0*1/tile_width]) #x(s') components. have 50 of them
    gradient_indices = features_old


    error = None
    v_old = 0.0
    for i in features_old:
        v_old+=w[i]

    
    #since curr_state is terminal, it has value of 0
    error = alpha*(reward-v_old)
       

    for i in gradient_indices:
        w[i] = w[i]+error

    return


def agent_cleanup():
    """
    This function is not used
    """
    # clean up
          
        
    return


def agent_message(in_message): # returns string, in_message: string
    global Q
    """
    Arguments: in_message: string
    returns: The value function as a string.
    This function is complete. You do not need to add code here.
    """
    # should not need to modify this function. Modify at your own risk
    if (in_message == 'ValueFunction'):
        return pickle.dumps(np.max(Q, axis=1), protocol=0)
    elif(in_message== "get RMSE"):
        # return 0
        rmse = get_RMSE()
        # print 'rmse', rmse
        
        return rmse
    else:
        return "I don't know what to return!!"

if __name__ == '__main__': 
    x = np.load("TrueValueFunction.npy")