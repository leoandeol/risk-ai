from gym_risk.envs import RiskEnv
import gym
from random import choice

env = gym.make('DraftingRisk-v0')
obs = env.reset()
done = False
while not done:
    empty = [x for x, y in obs[1].owners.items() if y is None]
    if not len(empty)==0:
        action = choice(empty)
        obs, reward, done, info = env.step(action)
    else:
        mine = [x for x, y in obs[1].owners.items() if y is not None and y.name == "Player"]
        action = choice(mine)
        obs, reward, done, info = env.step(action)

print("reward", reward)
