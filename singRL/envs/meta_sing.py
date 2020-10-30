import math
import numpy as np

import TransportMaps as TM
import TransportMaps.Algorithms.SparsityIdentification as SIQ

import gym
from gym import spaces
from gym.utils import seeding

class SINGEnv(gym.Env):

    def __init__(self):
        self.p_bounds = np.array([4,8]) # min and max graph dimension
        self.n_bounds = np.array([500,5000]) # min and max samples
        self.iter = 0
        self.iter_max = 100

        self.action_space= spaces.MultiDiscrete([3,10,1]) # Beta,delta,ordering

        self.seed()
        self.reset()
        self.observation_space = spaces.MultiBinary([self.p,self.p])
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # Map action to SING input variables
        order = action[0] + 1
        delta = action[1] + 1
        ordering = SIQ.ReverseCholesky()
        # if action[2] is 0:
        #     ordering = SI.ReverseCholesky()

        # Evaluate SING given action
        generalized_precision = SIQ.SING(self.data, order, ordering, delta)
        graph_result = np.greater(generalized_precision,0)*1

        ## Compute reward and if done
        # 

        # Is graph perfect?
        if (self.true_graph==graph_result).all():
            done = True
            reward = 100

        # If graph is not perfect
        else:
            # case = np.array([0, 0])
            done = False
            case1,case2 = 0,0

            for i in range(0,self.p):
                for j in range(i+1,self.p):

                    if not graph_result[i,j]==self.true_graph[i,j]:

                        if self.true_graph[i,j]:
                            case1 = case1 + 1
                        else:
                            case2 = case2 + 1

            reward = 80*(math.exp(-2*case1/3)*math.exp(-case2/3) -order/85)
            

        self.iter += 1
        if self.iter > self.iter_max:
            done = True

        return graph_result, reward, done, generalized_precision

    def reset(self):
        # Dimensionality
        self.p = self.np_random.randint(self.p_bounds[0],self.p_bounds[1]+1)
        # Sample size
        self.n = self.np_random.randint(self.n_bounds[0],self.n_bounds[1]+1)

        # Generate precision matrix
        sig_inv = np.identity(self.p)

        for i in range(0,self.p):
            for j in range(i+1,self.p):

                # Introduce sparcity
                if self.np_random.uniform(0,1)>0.5:
                    sig_inv[i,j] = self.np_random.uniform(0.25,0.5)
                    sig_inv[j,i] = sig_inv[i,j]
        
        # Compute covariance matrix
        cov = np.linalg.inv(sig_inv)

        # Generate data
        self.data = self.np_random.multivariate_normal(np.zeros(self.p),cov,self.n)

        # Declare true graph
        self.true_graph = np.greater(sig_inv,0)*1
