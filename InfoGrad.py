
import numpy as np
from jax import numpy as jnp
from jax import jit, grad,vmap

vdot = jit(vmap(jnp.dot,in_axes=[0,1])) # [a_i] = W[i,:].dot(M[:,i])
        
class InfoGrad(object):
    """
    The class containing calculation utilities of information gradient.
    Input:
            k: the kernel function. Must be a jax differentiable function.
    """
    def __init__(self, k):
        
        self.k = k

        dkElement = jit(vmap(grad(k,argnums=0),in_axes=[None,0,None,None]))
        self.dk = jit(vmap(dkElement,in_axes=[0,None,None,None])) # dk=[dk/dx1], with dk(x,X_{1:n}) vector calculation enabled.
        self.v_k = jit(vmap(k,in_axes=[0,None,None,None]))

    def dIdx(self,model,curr_x,X):
        '''
            The vectorization needs further work, for the algorithm to work properly on any space_dim.
            Currently, the dk and v_k are very slow, which should be optimized in the future.
        '''

        assert(len(curr_x.shape)>1)
        assert(len(X.shape)>1)

        m,space_dim = curr_x.shape

        c = model.kernel_.get_params()['k1__constant_value']

        l = model.kernel_.get_params()['k2__length_scale']

        _,K_t = model.predict(curr_x,return_cov=True)
#         K_t = np.eye(len(curr_x))

        M = np.linalg.inv(np.eye(len(K_t))+K_t/model.alpha)
#         M = K_t


        S_n = self.v_k(X,X,c,l)
#         S_n = np.eye(len(X))

        s = model.alpha


        L = self.v_k(curr_x,X,c,l)
#         L = np.ones((len(curr_x),len(X)))
#         print(L.shape)

        # print(self.dk(curr_x,X,c,l).reshape(-1,len(X)))
        # print(np.linalg.inv(S_n).dot(L.T))

        W = self.dk(curr_x,curr_x,c,l).reshape(-1,len(curr_x))
#         print(W.shape)
#         W = np.ones((2*len(curr_x),len(curr_x)))
        
        A = self.dk(curr_x,X,c,l).reshape(-1,len(X)) 
#         print(A.shape)
#         A = np.ones((2*len(curr_x),len(X)))
        W-= A.dot(np.linalg.inv(S_n+s*np.eye(len(S_n)))).dot(L.T)
        
        W = W.reshape((m,space_dim,m))
#         W  = np.ones((m,space_dim,m))

        # print(W.shape,M.shape)
        
        return vdot(W,M)

#         return np.ones(curr_x.shape)
        # pass