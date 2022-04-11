import numpy as np
import cvxpy as cp
from jax import grad,jit
from jax import numpy as jp

from RKHS import Gram


def rand_search(x0,R,objective,T,step_size):
    xss = []
    vals = []
    for _ in range(50000):
        
        xs = [x0]
        for i in range(T-1):
            
            cand_x = 0
            while True:
                r = step_size

                theta = np.random.rand()*2*np.pi

                cand_x = xs[-1]+r*np.array([np.cos(theta),np.sin(theta)])

                if np.linalg.norm(cand_x-x0)<=R:
                    break

            xs.append(cand_x)

        xs = np.array(xs)
        val = objective(xs)
        
        xss.append(xs)
        vals.append(val)
            
    return xss[np.argmax(vals)],np.max(vals),xss,vals

def rand_u_search(x0,R,x_objective,T,step_size):
   
    def trajectory(x1,u):
        '''
        Generate x[1:T] given x[1] and u[1:T-1]
        '''
        return x1+np.cumsum(np.vstack([np.zeros(u.shape[-1]),u]),axis=0)

    objective = lambda x1,u: x_objective(trajectory(x1,u))
    uss = []
    vals = []
    for _ in range(50000):
        
        xs = [x0]
        us = []
        for i in range(T-1):
            
            cand_x = 0
            cand_u = 0
            while True:
                r = step_size

                theta = np.random.rand()*2*np.pi
                
                cand_u = r*np.array([np.cos(theta),np.sin(theta)])
                
                cand_x = xs[-1]+cand_u

                if np.linalg.norm(cand_x-x0)<=R:
                    break

            xs.append(cand_x)
            us.append(cand_u)

        us = np.array(us)
        val = objective(x0,us)
        
        uss.append(us)
        vals.append(val)
            
    return uss[np.argmax(vals)],np.max(vals),uss,vals

def incremental_greedy(kernel,x0,step_size,ref,R,T,var_0,c):
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
    xs = [x0]

    n_test = 2 * 10 ** 5

    for loc in range(T-1):
        # Generate the feasible random samples
        rand_theta = np.random.rand(n_test)*2*np.pi

        rand_dir = np.array([np.cos(rand_theta),np.sin(rand_theta)])

        x_test = xs[-1]+(step_size*np.random.rand(n_test)*rand_dir).T # Step size constraint.

        x_test = x_test[(np.linalg.norm(x_test[:,np.newaxis,:]-ref,axis=-1)<=R[loc+1,:]).ravel()] # Bounded region constraint.

        if len(x_test)>0:# If some x_test are feasible.
            gain = marginal_gain(x_test,np.array(xs),kernel)

            x_best = x_test[np.argmax(gain)]
        else: # If none of the x_tests are feasible(the region is infeasible)
            x_best = xs[-1]

        xs.append(x_best)
    return np.array(xs)

def projected_x_gradient(objective,initial_states,ref,R,T,step_size):
    n_iter = 100
    delta_tolerance = 1e-7 # If ||z_{t+1}-z_t||<delta_tolerance, terminate the algorithm.
    eta = 0.01 # The learning rate for gradient update

    pg_zs = []
    
    def project(z,ref,R,step_size):
        T = len(z)
        N = len(ref)
        # Setup the projection optimization problem

        z_proj = cp.Variable(z.shape)

        # Step size constraints
        constraints = [cp.norm(z_proj[i+1]-z_proj[i])<=step_size for i in range(T-1)] 

        # # Bounded search region constraints
        constraints += [cp.norm(z_proj[i]-ref[j])<=R[i,j] for i in range(T) for j in range(N)]

        prob = cp.Problem(cp.Minimize(cp.norm(z_proj-z)),constraints)

        prob.solve()

        return z_proj.value

    for k,z_0 in enumerate(initial_states):

        print('{}th start outof {}.'.format(k,len(initial_states)))

        # z_0 = np.array([np.arange(T)*step_size,np.zeros(T)]).T*step_size # set z_0 to be the straight line

        z = np.array(z_0)

        g = jit(grad(objective))
        
        best_z = z_0
        best_val = objective(z_0)

        for _ in range(n_iter):
            z += eta * g(z) # Gradient step
            z = project(z,ref,R,step_size) # Projection
            if objective(z)> best_val:
                best_z = z

        pg_zs.append(best_z)
    

        ob = [objective(z) for z in pg_zs]

        return pg_zs[np.argmax(ob)]

def projected_u_gradient(x_objective,x0,initial_us,ref,R,step_size):
    def trajectory(x1,u):
        '''
        Generate x[1:T] given x[1] and u[1:T-1]
        '''
        return x1+jp.cumsum(jp.vstack([np.zeros(u.shape[-1]),u]),axis=0)

    def project(x0,u,ref,R,step_size):
        N = len(ref)
        T = len(u)
        # Setup the projection optimization problem

        z = x0+np.cumsum(np.vstack([np.zeros(u.shape[-1]),u]),axis=0)

        u_proj = cp.Variable(u.shape)

        z_proj = x0+cp.cumsum(cp.vstack([np.zeros(u.shape[-1]).reshape(1,-1),u_proj]),axis = 0)

        # Step size constraints
        constraints = [cp.norm(u_proj,axis=1)<=step_size] 

        # # Bounded search region constraints
        constraints += [cp.norm(z_proj[i]-ref[j])<=R[i,j] for i in range(T) for j in range(N)]

        prob = cp.Problem(cp.Minimize(cp.norm(z_proj-z)),constraints)

        prob.solve()

        return u_proj.value

    objective = jit(lambda x1,u: x_objective(trajectory(x1,u)))

    
    n_iter = 100
    delta_tolerance = 1e-7 # If ||z_{t+1}-z_t||<delta_tolerance, terminate the algorithm.
    eta = 0.01 # The learning rate for gradient update

    pg_us = []

    g = jit(grad(objective,argnums=[0,1]))

    for m,u_0 in enumerate(initial_us):

        print('{}th start outof {}.'.format(m,len(initial_us)))


        # z_0 = np.array([np.arange(T)*step_size,np.zeros(T)]).T*step_size # set z_0 to be the straight line

        u = np.array(u_0)

        best_u = u_0
        best_val = objective(x0,u_0)

        for _ in range(n_iter):
            dx1,du=g(x0,u)
            u += eta * du # Gradient step
            u = project(x0,u,ref,R,step_size) # Projection

 
            if objective(x0,u)> best_val:
                best_u = u
                best_val = objective(x0,u)

        pg_us.append(best_u)
        
    ob = [objective(x0,u) for u in pg_us]

    return np.array(trajectory(x0,pg_us[np.argmax(ob)]))