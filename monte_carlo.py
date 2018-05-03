import random
import easy21
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np

Nsa = dict()
Ns = dict()
Qsa = dict()
k = 1


def Q(s, a):
    return Qsa.get((s,a), 0)


def N(s, a=None):
    if a is None:
        return Ns.get(s, 0)
    else:
        return Nsa.get((s,a), 0)


def update_N(s, a):
    Nsa[(s,a)] = N(s,a) + 1
    Ns[s] = N(s) + 1


def get_e_greedy(s):
    e = 100/(100 + N(s))
    if random.random() > e:
        # Greedy action:
        qs_hit = Q(s, easy21.ACTION_HIT)
        qs_stick = Q(s, easy21.ACTION_STICK)
        if qs_stick > qs_hit:
            return easy21.ACTION_STICK
        else:
            return easy21.ACTION_HIT
    else:
        return random.randint(0, 1)


def run_episode(env : easy21.Easy21Env, k):
    """
    Plays an episode according to the e-greedy policy
    :param env:
    :param k:
    :return:
    """
    sa_list = []
    r_list = []
    s = env.get_state()
    while not env.terminated:
        a = get_e_greedy(s)
        sa_list.append((s, a))
        s, r = env.step(a)
        r_list.append(r)
    return sa_list, r_list


def evaluate_policy(sa_list, r_list):

    # Assume discount factor of 1:
    G = sum(r_list)
    for s, a in sa_list:
        update_N(s, a)
        Qsa[(s, a)] = Q(s, a) + (G - Q(s, a))/(N(s, a))


def plot_V(pl, xx, yy, surf):
    player_sums = np.arange(0, 22)
    dealer_showing = np.arange(0, 11)
    V = np.zeros((player_sums.shape[0], dealer_showing.shape[0], 2))
    for su in player_sums:
        for de in dealer_showing:
            V[su, de, :] = np.array([
                Q((su, de, False), easy21.ACTION_HIT),
                Q((su, de, False), easy21.ACTION_STICK)])

    #pl.imshow(np.concatenate((V[:,:,0], V[:,:,1]), axis=0), interpolation='none')
    #pl.imshow(V[:,:,0], interpolation='none')
    if surf is not None:
        surf.remove()

    return pl.plot_surface(xx, yy, np.max(V[:,:,0:2], axis=2), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(0, 22)
y = np.arange(0, 11)
xx, yy = np.meshgrid(y, x)

surf = None
env = easy21.Easy21Env()
avg_score = 0
print()
while True:
    env.reset()
    sa_l, r_l = run_episode(env, k)
    evaluate_policy(sa_l, r_l)

    # Keep "avg"_score for fun and debugging:
    score = sum(r_l)
    avg_score = avg_score * 0.9999 + score * 0.0001

    if k%1000 == 0:
        print("Running avg: %.2f" % avg_score)
        surf = plot_V(ax ,xx, yy, surf)
        plt.draw()
        plt.pause(0.05)

    # Increment k
    k += 1


