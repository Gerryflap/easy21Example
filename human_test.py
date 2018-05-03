import easy21

env = easy21.Easy21Env()
print("Current state: <playersum: %d, dealercard: %s, game over: %s>"%env.get_state())
r = None
while not env.terminated:
    action = -1
    while action not in easy21.action_space:
        action = int(input("What action do you want to perform? 0 = hit, 1 = stick: "))
    s, r = env.step(action)
    print("Current state: <playersum: %d, dealercard: %s, game over: %s>"%s)

print(r)
