import numpy as np
from utils import RandomUnifBall

def x_single_improve(x,loc,x_objective,step_size,ref,R,x_root=None):
    '''
        Alter a single element of x, x[loc], for potential improvement in x_objective.
    
        x_root is required if loc==0.
        
        x_objective(x):
            x.shape = (n_x,T,space_dim)
            Output shape = (n_x,), output[i] = mutual information for x[i].

    '''
    n_test = 50000
    ### Generate candidate x solutions ###
    
    if loc==0:
        assert(not x_root is None)
        x_loc = RandomUnifBall(step_size,n_test,center = x_root) 
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
        x_best = x
        
    return x_best

def x_local_improve(x0,x_objective,step_size,ref,R,x_root,reverse_order = False):
    '''
        Call x_single_improve sequentially to obtain a local improvement of x0.
        
        
        x_objective(x):
            x.shape = (n_x,T,space_dim)
            Output shape = (n_x,), output[i] = mutual information for x[i].

    '''
    x = np.array(x0)
    
    for i in range(len(x)):
        if reverse_order:
            idx = (len(x)-i-1) % len(x)
        else:
            idx = i % len(x)
            
        x_best = x_single_improve(x,idx,x_objective,step_size,ref,R,x_root = x_root if idx%len(x)==0 else None)
        x = x_best
    
    return x
    
    