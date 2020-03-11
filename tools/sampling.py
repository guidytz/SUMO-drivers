'''
Created on 06/08/2014

@author: Gabriel de O. Ramos <goramos@inf.ufrgs.br>
'''
import random

#return the index of the greatest/smallest value in the list.
#if there are multiple items with the same value, select one 
#of them uniformly at random, using reservoir sampling
def reservoir_sampling(listv, greatest=True):
    sv = float("-inf")
    if (not greatest):
        sv = float("inf")
    si = -1
    
    n = 0
    for i, v in enumerate(listv):
        if (greatest and v > sv) or (not greatest and v < sv):
            sv = v
            si = i
            n = 1
        elif v == sv:
            n += 1
            if random.random() < 1.0/n:
                si = i
    
    return si

#return the index of one of the list elements. The elements in list
#represent the probability of their selection, i.e., it is a
#probability vector.  
def probability_vector(listv, sumup=None):
    
    #generate a random number in the range <0,sum>.
    #if the sum was not provided, calculate it
    try:
        r = random.random() * sumup
    except:
        r = random.random() * sum(listv)
    
    #find the element corresponding to the random number
    for i, v in enumerate(listv):
        if r < v:
            return i
        r -= v
    
    #this is not supposed to happen
    return random.randint(0, len(listv)-1)

