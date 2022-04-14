import numpy as np
import numpy.linalg as la

def Gram(kernel,x):
    '''
        kernel: callable, kernel(x,x').
        x: shape = (n_x, space_dim)
        
        Output: KA.shape = (n_x,n_x)
    '''
    KA = kernel(x[:,np.newaxis,:],x)
    return KA

def feature(kernel,ref_points,x):
    '''
        ref_points: shape=(N,space_dim)
        x:shape = (T,space_dim)
        
        Expect ref_points is dense(N is large, typically N>>T) in the environment of x.
        
        Output: feat.shape=(T,N)
        
        If feat = feature(kernel,ref_points,x), then can check feat.T.dot(feat)~=Gram(kernel,x)
    '''
    K = Gram(kernel,ref_points)
#     l,U =la.eig(K)
#     M = np.diag(1/np.sqrt(l)).dot(U.T) # M.T.dot(M) = K^{-1} is satisfied.
    # Another idea of choosing M, not from the eigen decomp approach though.
    M = la.cholesky(la.inv(K)).T

    if len(x.shape)<=1: # If x is a single vector, x.shape = (space_dim,)
        return M.dot(kernel(ref_points,x)) # Output shape = (feature_dim(==len(ref_points)),)
    else: # If x is a list of vector, x.shape= (T,space_dim)
        return M.dot(kernel(ref_points[:,np.newaxis,:],x)) #Output shape = (T,feature_dim)
    

# The most commonly used Squred-Exponential Kernel, and its GPMI.
def h(r,c,l):
    return c * np.exp(-(r**2) / l**2)

# The differentiable kernel function with parameters c,l not filled.

def k(x1,x2,c,l):
    small_sig = 1e-10 # This is needed for numerical stability.
    return h(np.linalg.norm(x1-x2+small_sig,axis = -1),c,l)

def GPMI(x,c,l,var_0):
    '''
    The mutual information for a GP.
    Fully vectorized.
    
    x.shape = (n_x,T,space_dim) or (T,space_dim)
    
    Output shape = (n_x,), output[i] = mutual information for x[i].
    '''
    

    if len(x.shape)<=2:
        x = x.reshape(-1,2)
        KA = k(x[:,np.newaxis,:],x,c,l)
    elif len(x.shape)==3:
        KA = k(x[:,:,np.newaxis,:],x[:,np.newaxis,:,:],c,l)

    if var_0>0:
        return 1/2*np.log(np.linalg.det(np.eye(KA.shape[1])+KA/var_0))
    else:
        return 1/2*np.log(np.linalg.det(KA))