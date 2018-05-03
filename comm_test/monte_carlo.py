import random
import comm_test.communication_test as comm
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
    e = 10000/(10000 + N(s))
    if random.random() > e:
        # Greedy action:
        a = np.argmax([Q(s, a) for a in range(comm.COMM_0, comm.GUESS_BLUE + 1)])
        return a
    else:
        return random.randint(comm.COMM_0, comm.GUESS_BLUE)


def run_episode(env : comm.CommEnv, k):
    """
    Plays an episode according to the e-greedy policy
    :param env:
    :param k:
    :return:
    """
    sa_list = [[], []]
    r_list = [[], []]
    player = 1
    s = env.get_state(player)
    while not env.terminated:
        s = env.get_state(player)
        a = get_e_greedy(s)
        sa_list[player-1].append((s, a))
        _, r = env.step(a, player)
        r_list[player-1].append(r)
        player = (player % 2) + 1
    return sa_list, r_list


def evaluate_policy(sa_list, r_list):

    # Assume discount factor of 1:
    G = [0,0]
    G[0] = sum(r_list[0])
    G[1] = sum(r_list[1])
    for player_index in range(2):
        for s, a in sa_list[player_index]:
            update_N(s, a)
            Qsa[(s, a)] = Q(s, a) + (G[player_index] - Q(s,a))/(N(s, a))


env = comm.CommEnv(2)
avg_score = 0
print()
while True:
    env.reset()
    sa_l, r_l = run_episode(env, k)
    evaluate_policy(sa_l, r_l)

    # Keep "avg"_score for fun and debugging:
    score = sum(r_l[0]) + sum(r_l[1])
    avg_score = avg_score * 0.9999 + score * 0.0001

    if k%1000 == 0:
        print("Running avg: %.2f" % avg_score)

    # Increment k
    k += 1


