import json
import easy21
import sarsa_lambda
import matplotlib.pyplot as plt

with open("mc-qvalues.json", 'r') as f:
    Qsa_star_inp = json.loads(f.read())
    Qsa_star = dict()
    for key, value in Qsa_star_inp.items():
        splitted = key.split()
        sa = ((int(splitted[0]), int(splitted[1]), bool(splitted[2])), int(splitted[3]))
        Qsa_star[sa] = value
    print(Qsa_star)


def mse(qsa):
    s = 0
    n = 0
    for k in qsa.keys() | Qsa_star.keys():
        n += 1
        s += (qsa.get(k, 0) - Qsa_star.get(k, 0))**2
    return s/n

env = easy21.Easy21Env()
ys = []
ls = []
for l in [i/10 for i in range(11)]:
    y = 0
    ny = 0
    ls.append(l)
    for j in range(30):
        agent = sarsa_lambda.SarsaLambdaAgent(l, easy21.action_space)
        for i in range(1000):
            agent.run_episode(env)
        y += mse(agent.Qsa)
        ny += 1
    ys.append(y/ny)

plt.plot(ls, ys)
plt.show()