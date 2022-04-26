import numpy as np

from Algorithms import no_step_greedy_ball,incremental_greedy

import RKHS
from RKHS import Gram

import cvxpy as cp

from shapely.geometry import Point

from shapely.vectorized import contains

from utils import RandomUnifBall,verify_feasibility

# Reverse upperbound using step size removal

def reverse_upperbound_no_step(x_objective,kernel,greedy_sol,T):
    '''
        Calculate the reverse greedy upper bounds with no step constraints.
        Output: out[i] = upper bound on optimal objective value, if at most T=i samples is allowed.
    '''
    
    g = greedy_sol

    partial_objective = np.array([x_objective(g[:i]) for i in range(len(g)+1)])

    return np.array([np.min(partial_objective[1:]/(1-np.power(1-1/t,np.arange(1,len(greedy_sol)+1)))) for t in range(1,T+1)])


def reverse_upperbound_stepped(x_objective,kernel,greedy_sol,search_region_center,search_region_radius,c,var_0,step_size,T):
    '''
        A longer greedy_sol potentially leads to tighter upper bounds.
        
        It is recommended that len(greedy_sol)>=2*T.
    ''' 
    
    def marginal_gain(x_new,x_old,kernel):
        '''
        Calculate the marginal gain in mutual information, given x_old has been collected, and a list of candidate x_new's is to be collected.

        x_old.shape = (t,space_dim)

        x_new.shape = (n,space_dim)

        Output: shape = (n), output[i] = marginal gain for x_new[i].
        '''

        S = Gram(kernel,x_old)

        k_t = kernel(x_new[:,np.newaxis,:],x_old).T



        return 1/2*np.log(1 + c/var_0 \
                   - 1/var_0**2 *np.sum(k_t.T.dot(np.linalg.inv(\
                                np.eye(len(S))+1/var_0 * S))* k_t.T,axis=-1))
    n_test = 10000
    
    search_region = Point(search_region_center).buffer(search_region_radius)

    x_test = RandomUnifBall(search_region_radius,n_test,center = search_region_center)

    step_size_envelopes = [Point(search_region_center).buffer((i-1)*step_size) for i in range(2,T+1)]

    feasible_regions = [e.intersection(search_region) for e in step_size_envelopes]

    feasible_xs = [x_test[verify_feasibility(e,x_test)] for e in step_size_envelopes]
    
    upperbounds = [x_objective(greedy_sol[:l])+np.sum([np.max(marginal_gain(feasible_xs[i],greedy_sol[:l],kernel)) for i in range(len(feasible_xs))]) for l in range(1,len(greedy_sol))] 

    return np.min(upperbounds)


def relaxed_convex_upperbound(step_size,c,l,var_0,kernel,ref,R,T,search_region_radius):
    d = RKHS.h(step_size,c=c,l=l)

    A = Gram(kernel,ref)

    N = len(A)


    b = np.ones((T,N))*np.array(RKHS.h(R,c=c,l=l))
    b[0,0] = RKHS.h(0,c=c,l=l) # Initial location constraint.

    # Define and solve the problem
    B = cp.Variable((T,N))

    S = cp.Variable((T,T),symmetric=True)

    M = cp.vstack([cp.hstack([S,B]),
                   cp.hstack([B.T,A])])


    constraints = [M>>0]


    constraints += [cp.diag(S)==c]

    constraints += [S[i,i+1]>=d for i in range(0,T-2)]

    constraints += [S>=RKHS.h(2*search_region_radius,c,l)]

    constraints += [S[0,:]>=b.flatten(),S[:,0]>=b.flatten()]


    prob = cp.Problem(cp.Maximize(1/2*cp.log_det(np.eye(T)+ S/var_0)),constraints)

    upper_bound = prob.solve()
    
    return upper_bound
