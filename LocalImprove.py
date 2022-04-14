import numpy as np
from utils import RandomUnifBall

import warnings

def x_single_improve(x,loc,x_objective,step_size,ref,R,x_root=None):
    '''
        Alter a single element of x, x[loc], for potential improvement in x_objective.
    
        x_root, the center of sampling ball for x[0], is required if loc==0.

        ref.shape = (N,space_dim)
        R.shape = (T,N)
        
        x_objective(x):
            x.shape = (n_x,T,space_dim)
            Output shape = (n_x,), output[i] = mutual information for x[i].

    '''
    n_test = 50000
    ### Generate candidate x solutions ###
    
    if loc==0:
        assert(not x_root is None)
        x_loc = RandomUnifBall(step_size,n_test,center = x_root) 
        x_loc = np.vstack([x_root,x_loc])
    else:
        x_loc = RandomUnifBall(step_size,n_test,center = x[loc-1]) 
    # The x[loc] within x_1:T to be altered. 

    # Discard the x_loc's that violate search region constraints.
    x_loc = x_loc[(np.linalg.norm(x_loc[:,np.newaxis,:]-ref,axis=-1)<=R[loc,:]).ravel()]

    if loc<len(x)-1:
        # Discard the x_loc's that violate the step size constraint with x[loc+1]
        x_loc = x_loc[(np.linalg.norm(x_loc[:,np.newaxis,:]-x[loc+1],axis=-1)<=step_size).ravel()]
    ######################################
  
    if len(x_loc)>0: # If all the x_loc's are feasible.
        x_cand = np.array([x for _ in range(len(x_loc))]) 
        
        x_cand[:,loc,:] = x_loc
       
        # x itself should be one of the candidates, to ensure a guaranteed improvement.
        x_cand = np.concatenate([x_cand,x[np.newaxis,:,:]])
       
        best_cand = np.argmax(x_objective(x_cand))

        x_best = x_cand[best_cand]
    else:
        warnings.warn('No feasible solution is found for loc={}.'.format(loc))
        x_best = x
        
    return x_best

def x_local_improve(x0,x_objective,step_size,ref,R,x_root=None,n_pass=1,reverse_order = False):
    '''
        Call x_single_improve sequentially to obtain local improvements of x0.
        
        x_root, the center of sampling ball for x[0]. If x_root is None, x[0] will be fixed.

        ref.shape = (N,space_dim)
        R.shape = (T,N)


        n_pass: the number of full-length passes to be run. 
                A full-length pass is equivalent as calling x_single_improve for len(x) times.

        
        x_objective(x):
            x.shape = (n_x,T,space_dim)
            Output shape = (n_x,), output[i] = mutual information for x[i].

    '''
    x = np.array(x0)

    for _ in range(n_pass):
        for i in range(len(x)):
            if reverse_order:
                idx = (len(x)-i-1) % len(x)
            else:
                idx = i % len(x)
                
            if x_root is None and idx==0:
                continue
            else:
                x_best = x_single_improve(x,idx,x_objective,step_size,ref,R,x_root = x_root if idx%len(x)==0 else None)
                x = x_best
    
    return x
    


def u_single_improve(x,loc,x_objective,step_size,ref,R):
    '''
    Alter a single u[loc]=x[loc+1]-x[loc], for potential improvement in x_objective.

    ref.shape = (N,space_dim)
    R.shape = (T,N)

    x_objective(x):{
        x.shape = (n_x,T,space_dim)
        Output shape = (n_x,), output[i] = mutual information for x[i].}

    '''
    def trajectory(x1,u):
        '''
        x1.shape = (n,dim) or (dim)
        u.shape = (n,T-1,dim)
        
        Generate x[n,1:T,dim] given x1 and u[n,1:T-1,dim]
        
        '''    
        u_pad = np.pad(u,((0,0),(1,0),(0,0)),'constant',constant_values = 0)
        u_pad[:,0,:] = x1
        return np.cumsum(u_pad,axis=1)

    def traj_to_u(x):
        '''
            u_t = x_t - x_{t-1}
            
            x.shape = (n_x,T,x_dim)
            
            output: u. u.shape = (n_x,T-1,x_dim)
        '''
        
        return  x[:,1:,:]-x[:,:-1,:]
    
    n_cand = 50000
    
    u0 = traj_to_u(np.array([x]))[0]

    u_loc = RandomUnifBall(step_size,n_cand)

    u_cand = np.array([u0 for _ in range(len(u_loc))]) 

    u_cand[:,loc,:] = u_loc

    x_cand = trajectory(x[0],u_cand)

    # x itself should be one of the candidates, to ensure a guaranteed improvement.

    x_cand = np.concatenate([x_cand,x[np.newaxis,:,:]])


    # Discard the x_cand's that violate search region constraints.
    x_cand = x_cand[(np.linalg.norm(x_cand-ref,axis=-1)<=R.T).all(axis=1)]

    if len(x_cand)>0:

        best_cand = np.argmax(x_objective(x_cand))

        return x_cand[best_cand]
    else:  
        # print('No feasible')
        warnings.warn('No feasible solution is found at loc {}.'.format(loc))
        return x
    
def u_local_improve(x0,x_objective,step_size,ref,R,n_pass=1,x_root=None):
    '''
        Call u_single_improve sequentially to obtain a local improvement of x0.
        

        ref.shape = (N,space_dim)
        R.shape = (T,N)
        
        x_objective(x):
            x.shape = (n_x,T,space_dim)
            Output shape = (n_x,), output[i] = mutual information for x[i].

        n_pass: the number of full-length passes to be run. 
                    A full-length pass is equivalent as calling u_single_improve for len(x)-1 times.
    '''

    x = np.array(x0)

    for _ in range(n_pass):
        # Improve the root of the trajectory(if applicable.)
        if not x_root is None:
            x = x_single_improve(x,0,x_objective,step_size,ref,R,x_root=x_root)

        for loc in range(len(x)-1):
            x = u_single_improve(x,loc,x_objective,step_size,ref,R)
    
    return x