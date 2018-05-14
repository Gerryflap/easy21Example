import easy21
import sarsa_lambda_function_approx as sarsa_lambda
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import tensorflow as tf


def f(inp):
    """
    Inp is a 4 float tensor consisting of sum of cards, dealer card, terminal state (0 or 1) and action (0 or 1)
    :param inp:
    :return: Qsa
    """
    inp = tf.reshape(inp, (1,4))

    W0, B0 = tf.Variable(tf.random_normal((3, 1))), tf.Variable(tf.random_normal((1,)))
    W1, B1 = tf.Variable(tf.random_normal((3, 1))), tf.Variable(tf.random_normal((1,)))

    f0 = lambda: tf.add(tf.matmul(inp[:, :-1], W0), B0)
    f1 = lambda: tf.add(tf.matmul(inp[:, :-1], W1), B1)
    return tf.reshape(tf.cond(tf.equal(inp[0, 3],0), f0, f1), tuple())


def f2(inp):
    inp = tf.reshape(inp, (1,4))
    x = tf.keras.layers.Dense(10, activation='tanh')(inp)
    x = tf.keras.layers.Dense(10, activation='tanh')(x)
    x = tf.keras.layers.Dense(10, activation='tanh')(x)
    x = tf.keras.layers.Dense(1, activation='tanh')(x)
    return tf.reshape(x, tuple())

def transform_sa(s, a):
    ((ps, dc, term), a) = (s,a)
    return np.array([ps, dc, float(term), a])



env = easy21.Easy21Env()

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(0, 22)
y = np.arange(0, 11)
xx, yy = np.meshgrid(y, x)


def plot_V(pl, xx, yy, surf, agent, sess):
    player_sums = np.arange(0, 22)
    dealer_showing = np.arange(0, 11)
    V = np.zeros((player_sums.shape[0], dealer_showing.shape[0], 2))
    for su in player_sums:
        for de in dealer_showing:
            V[su, de, :] = np.array([
                agent.Q((su, de, False), easy21.ACTION_HIT, sess),
                agent.Q((su, de, False), easy21.ACTION_STICK, sess)])

    #pl.imshow(np.concatenate((V[:,:,0], V[:,:,1]), axis=0), interpolation='none')
    #pl.imshow(V[:,:,0], interpolation='none')
    if surf is not None:
        surf.remove()

    return pl.plot_surface(xx, yy, np.max(V[:,:,0:2], axis=2), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)


avg_score = 0
k = 0
surf = None
with tf.Session() as sess:
    agent = sarsa_lambda.SarsaLambdaAgent(1, easy21.action_space, f2, (4,), gamma=1, N0=100, sa_transformer=transform_sa, alpha=0.0001)
    init = tf.global_variables_initializer()
    sess.run(init)
    while True:
        score = agent.run_episode(env, sess)
        avg_score = avg_score*0.999 + score*0.001
        if k%1000 == 0:
            print(avg_score)
            surf = plot_V(ax ,xx, yy, surf, agent, sess)
            plt.draw()
            plt.pause(0.05)
        k += 1

