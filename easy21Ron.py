import random

p_red = 1 / 3
p_black = 1 - p_red


def draw_card_value():
    return random.randint(1, 10)


def draw_card_color():
    return random.random() > p_red


def draw_card():
    return draw_card_value(), draw_card_color()


def draw_black_card():
    return draw_card_value(), True


def draw_init_values():
    return card_value(draw_black_card()), card_value(draw_black_card())


def draw_init_state():
    return draw_init_values() + (False,)


def card_value(card):
    v, c = card
    return v if c else -v


def valid_score(score):
    return 1 <= score <= 21


def reward(state):
    p_sum, d_sum, terminal = state
    if not terminal:
        return 0
    if not valid_score(p_sum):
        return -1
    if not valid_score(d_sum):
        return 1
    if p_sum > d_sum:
        return 1
    if p_sum < d_sum:
        return -1
    return 0


def step(s: tuple, a: bool):
    p_sum, d_sum, terminal = s
    if terminal:
        raise Exception('Cannot perform action on terminal state!')
    if a:  # Hit
        p_sum += card_value(draw_card())
        s = (p_sum, d_sum, not valid_score(p_sum))
    else:  # Stick -> play dealer
        while valid_score(d_sum) and d_sum < 17:
            d_sum += card_value(draw_card())
        s = (p_sum, d_sum, True)
    return s, reward(s)

class Easy21Env(object):
    def __init__(self):
        self.terminated = True
        self.s = draw_init_state()

    def reset(self):
        self.s = draw_init_state()
        self.terminated = False

    def step(self, a):
        sp, r = step(self.s, a)
        self.s = sp
        _, _, self.terminated = sp
        return sp, r