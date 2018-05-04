import random

ACTION_HIT = 0
ACTION_STICK = 1
action_space = {ACTION_HIT ,ACTION_STICK}


class Easy21Env(object):
    def __init__(self):
        self.playersum = 0
        self.dealershowing = 0
        self.terminated = True
        self.reset()

    def draw_card(self, red=None):
        if red is None:
            red = random.random() > 2/3
        value = random.randint(1, 10)
        if red:
            return -value
        else:
            return value

    def reset(self):
        self.playersum = self.draw_card(red=False)
        self.dealershowing = self.draw_card(red=False)
        self.terminated = False
        return self.get_state()

    def get_state(self):
        return self.playersum, self.dealershowing, self.terminated

    def step(self, action):
        if self.terminated:
            raise EnvironmentError("Tried stepping in terminated markov process")
        reward = 0
        if action == ACTION_HIT:
            self.playersum += self.draw_card()
            if self.playersum > 21 or self.playersum < 1:
                self.terminated = True
                reward = -1
                return self.get_state(), reward
            return self.get_state(), reward
        else:

            # The dealer plays
            dealersum = self.dealershowing
            while True:
                if dealersum >= 17:
                    if dealersum > self.playersum:
                        reward = -1
                    elif dealersum == self.playersum:
                        reward = 0
                    else:
                        reward = 1
                    self.terminated = True
                    return self.get_state(), reward
                dealersum += self.draw_card()
                if dealersum > 21 or dealersum < 1:
                    # Dealer goes bust
                    self.terminated = True
                    reward = 1
                    return self.get_state(), reward
