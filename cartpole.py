import gym

import random
import numpy as np
import time

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
    e = 10/(10 + N(s))
    if random.random() > e:
        # Greedy action:
        a = np.argmax([Q(s, a) for a in range(2)])
        return a
    else:
        return random.randint(0,1)


def run_episode(env, k, draw=False):
    """
    Plays an episode according to the e-greedy policy
    :param env:
    :param k:
    :return:
    """
    sa_list = []
    r_list = []
    s = env.reset()
    done = False
    while not done:
        s = str(np.round(s, 0))
        a = get_e_greedy(s)
        sa_list.append((s, a))
        s, r, done, _ = env.step(a)
        if draw:
            env.render()
            time.sleep(1/60)
        r_list.append(r)
    return sa_list, r_list


def evaluate_policy(sa_list, r_list):

    # Assume discount factor of 1:
    G = sum(r_list)
    for s, a in sa_list:
        update_N(s, a)
        Qsa[(s, a)] = Q(s, a) + (G - Q(s,a))/(N(s, a))


env = gym.make('CartPole-v1')
avg_score = 0
print()
while True:
    sa_l, r_l = run_episode(env, k)
    evaluate_policy(sa_l, r_l)

    # Keep "avg"_score for fun and debugging:
    score = sum(r_l)
    avg_score = avg_score * 0.9999 + score * 0.0001

    if k%1000 == 0:
        print("Running avg: %.2f" % avg_score)
        run_episode(env, k, True)
    # Increment k
    k += 1




