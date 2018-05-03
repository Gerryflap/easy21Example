import random

COMM_0 = 0
COMM_1 = 1
COMM_ACTIONS = {COMM_0, COMM_1}
GUESS_RED = 2
GUESS_GREEN = 3
GUESS_BLUE = 4

COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2

action_space = {COMM_0, COMM_1, GUESS_RED, GUESS_GREEN, GUESS_BLUE}
COLORS = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]


class CommEnv(object):
    def __init__(self, max_length):
        self.p1_comms = None
        self.p2_comms = None
        self.p1_color = None
        self.p2_color = None
        self.terminated = True
        self.reset()
        self.max_length = max_length

    def reset(self):
        self.p1_comms = []
        self.p2_comms = []
        self.p1_color = random.choice(COLORS)
        self.p2_color = random.choice(COLORS)
        self.terminated = False

    def get_state(self, player):
        # Add random noise to the state to make random decisions possible
        noise = 0
        comms = self.p2_comms if player == 1 else self.p1_comms
        comms_own = self.p1_comms if player == 1 else self.p2_comms
        comm_str = str(comms)

        if player == 1:
            return comm_str, noise, self.p1_color, self.terminated
        else:
            return comm_str, noise, self.p2_color, self.terminated

    def step(self, action, player):
        if self.terminated:
            raise EnvironmentError("Tried stepping in terminated markov process")

        comms = self.p1_comms if player == 1 else self.p2_comms
        # The other color:
        np_color = self.p2_color if player == 1 else self.p1_color

        if action in COMM_ACTIONS:
            r = 0
            if action == COMM_0:
                comms.append(0)
                r = 0.0
            elif action == COMM_1:
                r = 0.0
                comms.append(1)
            if len(comms) > self.max_length:
                self.terminated = True
                return self.get_state(player), -2
            return self.get_state(player), r
        else:
            self.terminated = True
            if (action == GUESS_BLUE and np_color == COLOR_BLUE) \
                or (action == GUESS_RED and np_color == COLOR_RED) \
                or (action == GUESS_GREEN and np_color == COLOR_GREEN):

                return self.get_state(player), 2
            else:
                return self.get_state(player), -1
