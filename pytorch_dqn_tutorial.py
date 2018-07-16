import gym
import gym_risk
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from time import gmtime, strftime
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make("DraftingRisk-v0")

random.seed(1)
env.seed(1)

if "inline" in matplotlib.get_backend():
    from IPython import display

plt.ion()

moment = strftime("%Y-%m-%d--%H:%M:%S",gmtime())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(42*3, 42*4)
        self.linear2 = nn.Linear(42*4, 42*3)
        self.linear3 = nn.Linear(42*3, 42*2)
        self.linear4 = nn.Linear(42*2, 42)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 8

policy_net = DQN().to(device)
#target_net = DQN().to(device)
#target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()

optimizer = optim.Adam(policy_net.parameters(),lr=0.1)
loss_fn = nn.MSELoss()
memory = ReplayMemory(10000)
DATA_FOLDER = "data/"

steps_done = 0

def select_action(state,obs):
    global steps_done
    obs = obs[1].owners
    state = torch.FloatTensor(state)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done/EPS_DECAY)
    steps_done += 1
    #filter impossible cases
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax()
    else:
        return torch.FloatTensor([[random.choice([ i for i,v in zip(range(len(obs)),obs.values()) if v is None])]], device=device)

num_episodes = 1000
episode_durations = []
episode_reward = [0]*num_episodes
losses = []

def output_data():
    rew = open(DATA_FOLDER+"pytorch_reward"+moment, "w")
    for i, j in zip(range(len(episode_reward)),episode_reward):
        rew.write(str(i)+";"+str(j)+"\n")
    avg = sum(episode_reward)/len(episode_reward)
    rew.write("average = "+str(avg)+"\n")
    lr = LinearRegression()
    lr.fit(np.array(range(num_episodes)).reshape((-1,1)),episode_reward)
    rew.write(str(lr.coef_))
    rew.close()

    lo = open(DATA_FOLDER+"pytorch_losses"+moment,"w")
    for i, j in zip(range(len(losses)),losses):
        lo.write(str(i)+";"+str(j)+"\n")
    avg = sum(losses)/len(losses)
    lo.write("average = "+str(avg)+"\n")
    lr2 = LinearRegression()
    lr2.fit(np.array(range(len(losses))).reshape((-1,1)),losses)
    lo.write(str(lr2.coef_))
    lo.close()
    torch.save(policy_net, DATA_FOLDER+"policy_net"+moment+".pt")
    torch.save(target_net, DATA_FOLDER+"target_net"+moment+".pt")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state).view(128, -1)
    action_batch = torch.cat([x.view(1) for x in batch.action])
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch.float())
    state_action_values = state_action_values.gather(1, action_batch.view(-1,1))

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = policy_net(non_final_next_states.view(-1,126).float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def input_from_obs(obs):
    obs = obs[1]
    #first 42 is the player, the next 2*42 the two others
    first = [ 1 if x is not None and x.name == "Player" else 0 for x in obs.owners.values()]
    second = [ 1 if x is not None and x.name == "Random_1" else 0 for x in obs.owners.values()]
    third = [ 1 if x is not None and x.name == "Random_2" else 0 for x in obs.owners.values()]
    return list(first)+list(second)+list(third)

def id_to_territory(i,obs):
    return list(obs[1].owners.keys())[i]

for i_episode in range(num_episodes):
    print("===> Episode ",i_episode," : ")
    # Initialize the environment and state
    obs = env.reset()
    state = input_from_obs(obs)
    for t in count():
        # Select and perform an action
        action = select_action(state,obs)
        action_id = int(action.item())
        action = id_to_territory(action_id, obs)
        #print(str(action))
        obs, reward, done, _ = env.step(action)
        #reward /= 100
        episode_reward[i_episode] += reward
        if None not in obs[1].owners.values():
            while not done:
                obs, reward, done, _ = env.step(random.choice([ x for x, y in obs[1].owners.items() if y is not None and y.name is "Player"]))
                #reward /= 100
                episode_reward[i_episode] += reward
        reward = torch.tensor([int(reward)], device=device)

        # Observe new state
        if not done:
            next_state = input_from_obs(obs)
        else:
            next_state = None
            print("Reward : ",episode_reward[i_episode])

        # Store the transition in memory
        memory.push(torch.tensor(state) if state is not None else None,
                    torch.tensor(action_id),
                    torch.tensor(next_state) if next_state is not None else None,
                    reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    # Update the target network
    #if i_episode % TARGET_UPDATE == 0:
    #    target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
output_data()
