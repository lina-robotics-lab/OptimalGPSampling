import numpy as np
from numpy import linalg as la

def RandomUnifBall(r,n,center=0):
        '''
                Return n uniformly random samples from a 2D ball with radius r 
        '''
        rand_theta = np.random.rand(n)*2*np.pi

        rand_dir = np.array([np.cos(rand_theta),np.sin(rand_theta)])

        return center+(r*np.random.rand(n)*rand_dir).T # Step size constraint.
         
