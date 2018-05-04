import gym
import sarsa_lambda as sl
import numpy as np

def transform_state(s):
    s *= np.array([1, 1, 10, 1])
    s = np.round(s)
    return str(s)


env = sl.GymEnvWrapper(gym.make('CartPole-v0'), transform_state)
agent = sl.SarsaLambdaAgent(1, [0,1], gamma=1, N0=10)

avg_score = 0
k = 0
while True:
    score = agent.run_episode(env)
    avg_score = 0.99*avg_score + 0.01*score
    if k%100 == 0:
        print("Avg score: %.2f"%avg_score)
        env.set_rendering(True)
        agent.run_episode(env)
        env.set_rendering(False)
    k+=1
