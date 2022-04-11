import numpy as np

def substitution_multi_lateration(sensor_locs,rhat):
    '''
    sensor_locs: shape = (n_locs, space_dim)
    rhat: shape = (n_locs,)
    
    Find the single location determined by the known sensor locations(sensor_locs) and distance readings(rhat)
    
    The substituion method is used in finding the solution.
    '''
    
    A=2*(sensor_locs[-1,:]-sensor_locs[:-1,:])
    
    rfront=rhat[:-1]**2
    
    rback=rhat[-1]**2
    
    pback=np.sum(sensor_locs[-1,:]**2)
    
    pfront=np.sum(sensor_locs[:-1,:]**2,axis=1)
    
    B=rfront-rback+pback-pfront

    qhat=np.linalg.pinv(A).dot(B)

    return qhat

def iterative_multi_lateration(anchors,initial_guess,R,\
                                epsilon = 0.001,\
                                max_iter = 100):

    '''
        anchors: shape = (T,space_dim). The list of known locations.
        initial_guess: shape = (N,space_dim). The initial guess of unknown locations,
        R: The distance reading matrices. 

        The input format of R(important): Let X = vstack([anchors,initial_guess]), then R should satisfy R_ij = ||X_i-X_j||.

        Output: x_est. shape = initial_guess.shape
    '''

    T = len(anchors)
    N = len(initial_guess)

    assert(len(R)==T+N)


    lack = np.ones((T+N,T+N))-np.eye(T+N)

    # Different schemes of weight matrix M = [\sigma_{ij}^{-2}]_ij

    # M = np.ones((T,T))
    M = 1/((R+np.eye(T+N))**2)

    M*=lack


    MM = M+M.T
    MM*=lack
    M*=lack
 

    x_est =  initial_guess # The iterand.

    for _ in range(max_iter):
        
        X = np.vstack([anchors,x_est]) # Construct the total location vector X.


        ## (Do not use) The slow, for-loop implementation of iterative update ##

        # x_est_2=[]
        # for i in range(3,T):
        #     x_i_est = np.sum([(M[i,j]+M[j,i])*X[j,:]+\
        #                      (M[i,j]*R[i,j]+M[j,i]*R[j,i])*(X[i,:]-X[j,:])/np.linalg.norm(X[i,:]-X[j,:])\
        #                       for j in range(T) if not j==i],axis=0)\
        #                 /np.sum([M[i,j]+M[j,i] for j in range(T) if not j==i])
        #     x_est_2.append(x_i_est)

        # x_est_2 = np.array(x_est_2)

        ## (Do not use) The slow, for-loop implementation of iterative update ##



        ##### The vectorized implementation of iterative update #####


        D = np.linalg.norm(X[:,np.newaxis,:]-X,axis=-1)# D = [||X_i-X_j||]_ij
        D+=np.eye(len(D)) # This is to avoid division by zero error.
        
        MDR = (M*R+(M*R).T)/D

        X = (MM.dot(X)\
            +MDR.dot(np.ones(len(MDR))).reshape(-1,1)*X\
            -MDR.dot(X))\
        /np.sum(MM,axis=1).reshape(-1,1) # Denominator

        new_est = X[len(anchors):]
        
        diff = np.linalg.norm(x_est-new_est)

        x_est = new_est
        
        if diff<epsilon:
            break

        ##### The vectorized implementation of iterative update #####

            
        # Sanity check
        # print(x_est-x_est_2) 
    

    return x_est
