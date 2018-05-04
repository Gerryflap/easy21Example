import easy21
import sarsa_lambda
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

agent = sarsa_lambda.SarsaLambdaAgent(0, easy21.action_space, gamma=1, N0=100)
env = easy21.Easy21Env()

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(0, 22)
y = np.arange(0, 11)
xx, yy = np.meshgrid(y, x)

def plot_V(pl, xx, yy, surf, agent):
    player_sums = np.arange(0, 22)
    dealer_showing = np.arange(0, 11)
    V = np.zeros((player_sums.shape[0], dealer_showing.shape[0], 2))
    for su in player_sums:
        for de in dealer_showing:
            V[su, de, :] = np.array([
                agent.Q((su, de, False), easy21.ACTION_HIT),
                agent.Q((su, de, False), easy21.ACTION_STICK)])

    #pl.imshow(np.concatenate((V[:,:,0], V[:,:,1]), axis=0), interpolation='none')
    #pl.imshow(V[:,:,0], interpolation='none')
    if surf is not None:
        surf.remove()

    return pl.plot_surface(xx, yy, np.max(V[:,:,0:2], axis=2), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


avg_score = 0
k = 0
surf = None
while True:
    score = agent.run_episode(env)
    avg_score = avg_score*0.999 + score*0.001
    if k%10000 == 0:
        print(avg_score)
        surf = plot_V(ax ,xx, yy, surf, agent)
        plt.draw()
        plt.pause(0.05)
    k += 1

