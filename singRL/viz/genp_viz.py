import pickle
import numpy as np
import matplotlib.pyplot as plt

file_pi = open('genpArc.obj', 'rb')
genp = pickle.load(file_pi)
file_pi.close()


episodes = 4

plt.figure()
# episodes
for z in range(0,episodes):
    sz = genp[z].shape
    p = sz[0]
    timesteps = int(sz[1]/p)

    low = 0
    high = p
    # timesteps in episode
    for t in range(0,timesteps):
        for dim in range(0,p):
            if dim == 0:
                graph = genp[z][dim][low:high]
            else:
                graph = np.vstack((graph,genp[z][dim][low:high]))

        low = high
        high = high+p

        plt.imshow(graph)
        plt.colorbar()
        plt.pause(0.00001)
        plt.clf()

plt.show()
