import gym
import sarsa_lambda as sl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

def transform_state(s):
    s *= np.array([4, 50])
    s = np.round(s)
    return s[0], s[1]


env = sl.GymEnvWrapper(gym.make('MountainCar-v0'), transform_state)
agent = sl.SarsaLambdaAgent(1, [0,1,2], gamma=1, N0=1000)

avg_score = 0
k = 0
for i in range(2000):
    score = agent.run_episode(env)
    avg_score = 0.99*avg_score + 0.01*score
    if k%100 == 0:
        print("Avg score: %.2f"%avg_score)
        #env.set_rendering(True)
        #agent.run_episode(env)
        #env.set_rendering(False)
    k+=1

qs = np.array([[a,b] for ((a,b), c) in agent.Qsa.keys()])
qsa = np.zeros((qs.shape[0], qs.shape[0], 3))
xs = np.sort(qs[:, 0])
ys = np.sort(qs[:, 1])

xx, yy = np.meshgrid(xs, ys)
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        for a in [0,1,2]:
            qsa[i, j, a] = agent.Qsa.get(((x,y), a), 0)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("Position")
ax.set_ylabel("Speed")

ax.plot_surface(xx, yy, np.exp(qsa[:,:,2])/np.sum(np.exp(qsa), axis=2), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
plt.show()