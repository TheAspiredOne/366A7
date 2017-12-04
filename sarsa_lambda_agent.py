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
    global w, num_tilings, feature_vector,iht, tile_width

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


def perform3d():

    global w

    x = [-1.2]
    xdot = [-0.07]
    
    # building x, and xdot value lists.
    xstart = -1.2
    for i in range(49):
        xstart+=0.034
        x.append(xstart)

    dot_start = -0.07
    for i in range(49):
        dot_start+=0.003
        xdot.append(dot_start)

    ax = list()
    ay = list()
    q_list = list()
    for i in x:
        for j in xdot:
            max_q = -999999999.9
            for z in range(0,3):
                features = tiles(iht,num_tilings,[((i/(0.5+1.2))*(1/tile_width)), ((j/(0.07+0.07))*(1/tile_width))],[z]) #x(s) components. have 50 of them
                est_q = 0.0
                for ii in features:
                    est_q+=w[ii]
                if est_q > max_q:
                    max_q = est_q
            q_list.append(-1.0*max_q)
            ax.append(i)
            ay.append(j)


    np.save('x', ax)
    np.save('y',ay)
    np.save('z', q_list)
    return (ax,ay,q_list)






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
    action = random.randrange(0,3)
    

    old_state=state
    last_action=action


    return action


def agent_step(reward, state): # returns NumPy array, reward: floating point, this_observation: NumPy array
    """
    Arguments: reward: floting point, state: int between 1 and 1000
    Returns action as an int between -100 and 100
    """
    global  old_state, iht, curr_state, last_action, alpha, w, gamma, num_tilings, tile_width, epsilon, z, lambd
    
    # print old_state, state

    action = None
    curr_state = state
    err = reward
 
    #unpack the np arrays
    x,xdot = state
    o_x, o_xdot =  old_state

    

    
    #calculate indices of affected features of curr_state and old_state
    features_old = tiles(iht,num_tilings,[((o_x/(0.5+1.2))*(1/tile_width)), ((o_xdot/(0.07+0.07))*(1/tile_width))],[last_action]) #x(s) components. have 50 of them

    for i in features_old:
        err -= w[i] 
        z[i] = 1.0


    ######## ACTION SELECTION ########
    ExploreExploit = rand_un()
    if ExploreExploit < epsilon:
        action = random.randrange(0,3) # Explore
    else:

    ## GREEDY ACTION ##
        max_q  = -999999999.9 #approx -inf
        for i in range(0,3): #number of poss actions
            Xs = tiles(iht,num_tilings,[((x/(0.5+1.2))*(1/tile_width)), ((xdot/(0.07+0.07))*(1/tile_width))], [i]) #x(s') components. have 50 of them
            est_q = 0.0
            for j in Xs:
                est_q += w[j]

            if est_q > max_q:
                max_q = est_q
                action  =  i

    features_curr = tiles(iht,num_tilings,[((x/(0.5+1.2))*(1/tile_width)), ((xdot/(0.07+0.07))*(1/tile_width))], [action]) #x(s') components. have 50 of them
    for i in features_curr:
    	err += gamma*w[i]

    w += alpha*err*z
    z = z*gamma*lambd


    old_state = state
    last_action = action
    

    return action


def agent_end(reward):
    """
    Arguments: reward: floating point
    Returns: Nothing
    """
    global old_state, last_action, w, alpha, z, iht, num_tilings, tile_width

    err = reward

    o_x,o_xdot = old_state
    features_old = tiles(iht,num_tilings,[((o_x/(0.5+1.2))*(1/tile_width)), ((o_xdot/(0.07+0.07))*(1/tile_width))],[last_action]) #x(s) components. have 50 of them

    for i in features_old:
    	err -= w[i]
    	z[i] = 1.0

    w += alpha*err * z


    

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
    elif (in_message == 'perform3d'):
        threeD_data = perform3d()
        return threeD_data

    else:
        return "I don't know what to return!!"

if __name__ == '__main__': 
    x = np.load("TrueValueFunction.npy")