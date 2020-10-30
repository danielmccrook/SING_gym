import pickle
import numpy as np
import matplotlib.pyplot as plt


# open pickled object
file_pi = open('genpArc_50.obj', 'rb')

reading = True
z = 0

plt.figure()
plt.style.use("dark_background")

# episodes
while reading:
    try:
        genp = pickle.load(file_pi)

        # get episode traits
        sz = genp.shape
        p = sz[0]
        timesteps = int(sz[1]/p)

        # initialize bounds through timesteps
        low = 0
        high = p

        # timesteps in episode
        for t in range(0,timesteps):
            # include all dimensions
            for dim in range(0,p):
                if dim == 0:
                    # initialize graph
                    graph = genp[dim][low:high]
                else:
                    graph = np.vstack((graph,genp[dim][low:high]))

            # update timestep bounds
            low = high
            high = high+p

            # update figure
            plt.imshow(graph)
            plt.pause(0.0001)
            plt.clf()

        z = z+1
    except:
        reading = False

file_pi.close()
plt.show()
