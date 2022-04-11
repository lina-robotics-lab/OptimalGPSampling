import numpy as np
def honeycomb(R,N_layer):
    '''
        Generate the standard honeycomb lattice centered at the origin, with radius R and N_layer's.
    '''
    
    def hexagon(r):

        if r==0:
            hexagon = np.zeros(2)
        else:

            thetas = np.linspace(0,2*np.pi,7)[:-1]

            hexagon = r * np.array([np.cos(thetas),np.sin(thetas)]).T

        return hexagon

    def layer(r,layer_ind):
        if r == 0 or layer_ind == 0:
            return np.zeros((1,2))

        layer_base = hexagon(r)

        layer_perm = np.zeros(layer_base.shape)
        layer_perm[:-1] = layer_base[1:]
        layer_perm[-1] = layer_base[0]

        return np.vstack(np.linspace(layer_base,layer_perm,layer_ind+1)[:-1])

    return np.vstack([layer(R/(N_layer-1+1e-6)*i,i) for i in range(N_layer)])
