import sarsa_lambda_function_approx as srsfa
import easy21
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tensorflow as tf


def transform_sa(s, a):
    ps, dh, _ = s
    output = []
    # for d_lower, d_upper in [(1, 4), (4, 7), (7, 10)]:
    #     for s_lower, s_upper in [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]:
    for d_lower, d_upper in [(1, 3), (4, 7), (8, 10)]:
        for s_lower, s_upper in [(1, 5), (6, 10), (11, 15), (16, 18), (19, 20), (21, 22)]:
            for action in easy21.action_space:
                if d_lower <= dh <= d_upper and s_lower <= ps <= s_upper and action == a:
                    output.append(1)
                else:
                    output.append(0)
    return np.array(output)


def linear_approximation(x):
    parameters = tf.Variable(tf.random_normal((36,), 0, 0.000001))

    return tf.reduce_sum(x * parameters)

def plot_V(pl, xx, yy, surf, agent, sess):
    player_sums = np.arange(0, 22)
    dealer_showing = np.arange(0, 11)
    V = np.zeros((player_sums.shape[0], dealer_showing.shape[0], 2))
    for su in player_sums:
        for de in dealer_showing:
            for ac in [easy21.ACTION_HIT, easy21.ACTION_STICK]:
                V[su, de, :] = agent.Q((su, de, False), ac, sess)

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

agent = srsfa.SarsaLambdaAgent(1, easy21.action_space, linear_approximation, (36,), sa_transformer=transform_sa, epsilon=0.05, alpha=0.00001)

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