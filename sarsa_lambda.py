"""
    Implementation of Sarsa(λ)
"""
import random
import numpy as np


class SarsaLambdaAgent(object):
    def __init__(self, l, action_space, gamma = 1, N0 = 100):
        self.l = l
        self.Qsa = dict()
        self.Nsa = dict()
        self.Ns = dict()
        self.action_space = action_space
        self.N0 = N0
        self.gamma = gamma

    def Q(self, s, a):
        return self.Qsa.get((s,a), 0)

    def N(self, s, a=None):
        if a is None:
            return self.Ns.get(s, 0)
        else:
            return self.Nsa.get((s,a), 0)

    def count_N(self, s, a):
        self.Nsa[(s,a)] = self.Nsa.get((s,a), 0) + 1
        self.Ns[s] = self.Ns.get(s, 0) + 1

    def get_e_greedy(self, s):
        e = self.N0 / (self.N0 + self.N(s))
        if random.random() > e:
            # Greedy action:
            max_a = None
            max_Q = None
            for a, Q in [(a, self.Q(s, a)) for a in self.action_space]:
                if max_a is None or max_Q < Q:
                    max_a = a
                    max_Q = Q
            return max_a
        else:
            return random.choice(list(self.action_space))

    def run_episode(self, env):
        s = env.reset()
        a = self.get_e_greedy(s)

        E = dict()
        score = 0

        while not env.terminated:
            s_prime, r = env.step(a)
            a_prime = self.get_e_greedy(s_prime)
            delta = r + self.gamma * self.Q(s_prime, a_prime) - self.Q(s, a)
            E[(s,a)] = E.get((s,a), 0) + 1
            self.count_N(s,a)
            for s,a in E.keys():
                alpha = 1/self.N(s,a)
                self.Qsa[(s,a)] = self.Q(s,a) + alpha*delta*E[(s,a)]
                E[(s,a)] *= self.gamma * self.l
            s, a = s_prime, a_prime
            score += r
        return score
