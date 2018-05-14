import deep_q_learning
import easy21
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tensorflow as tf


def transform_s(s):
    a,b,_ = s

    s = np.zeros((22,))
    if 0 < a < 22:
        s[a-1] = 1
    dh = np.zeros((11,))
    dh[b-1] = 1
    return np.concatenate((s,dh))




def deep_net(x):
    ks = tf.keras
    x = ks.layers.Dense(300, activation='relu')(x)
    x = ks.layers.Dense(200, activation='relu')(x)
    x = ks.layers.Dense(100, activation='relu')(x)
    x = ks.layers.Dense(50, activation='relu')(x)
    x = ks.layers.Dense(2, activation='linear')(x)
    return x


def plot_V(pl, xx, yy, surf, agent, sess):
    player_sums = np.arange(0, 22)
    dealer_showing = np.arange(0, 11)
    V = np.zeros((player_sums.shape[0], dealer_showing.shape[0], 2))
    for su in player_sums:
        for de in dealer_showing:
                V[su, de, :] = agent.Q((su, de, False), sess)

    #pl.imshow(np.concatenate((V[:,:,0], V[:,:,1]), axis=0), interpolation='none')
    #pl.imshow(V[:,:,0], interpolation='none')
    if surf is not None:
        surf.remove()

    return pl.plot_surface(xx, yy, np.max(V[:,:,:], axis=2), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(0, 22)
y = np.arange(0, 11)
xx, yy = np.meshgrid(y, x)

agent = deep_q_learning.DeepQAgent(easy21.action_space, deep_net, (33,), state_transformer=transform_s, alpha=0.00001, fixed_steps=100)

env = easy21.Easy21Env()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    avg_score = 0
    k = 0
    surf = None

    while True:
        score = agent.run_episode(env, sess)
        avg_score = avg_score*0.999 + score*0.001
        if k%100 == 0:
            surf = plot_V(ax, xx, yy, surf, agent, sess)
            plt.draw()
            plt.pause(0.05)
            print(avg_score)
        k += 1