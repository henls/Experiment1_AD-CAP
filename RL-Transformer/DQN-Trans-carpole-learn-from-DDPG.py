from cProfile import label
import itertools
import re
from sys import prefix
from turtle import forward
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR
torch.autograd.set_detect_anomaly(True)

import numpy as np

from line_profiler import LineProfiler
############################################ init #################################
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
env = gym.make('CartPole-v0').unwrapped
env.seed(1000)
torch.manual_seed(200)
random.seed(60)
max_length = 10

BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
tau = 0.0005
LR_TS = 1e-3
LR_RL = 1e-3
N_STATES = 4
outputs_dim = env.action_space.n
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################ replay memory #################################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

################################ Sequence States #################################
class SequenceStates(object):
    def __init__(self, max_length):
        """
        *输入：： max_length:状态序列的长度
        *输出：： 当前管道中的状态
        """
        self.max_length = max_length
        self.stats_que = deque([], maxlen= max_length)

    def push(self, frame):
        self.stats_que.append(frame)

    def get(self):
        assert self.__len__() == self.max_length
        return torch.vstack([i for i in self.stats_que]).reshape(self.max_length, -1, len(self.stats_que[0]))
    
    def __len__(self):
        return len(self.stats_que)

############################################ Transformer #################################

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #单词在句子中的位置
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransAm(nn.Module):
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.5):
        super(TransAm, self).__init__()
        global max_length
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
                    nn.Linear(N_STATES,16), 
                    nn.ReLU(),
                    nn.Linear(16, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                    nn.ReLU()
                )

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(max_length).to(device)
            self.src_mask = mask
        src = self.fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class RlNet(nn.Module):
    def __init__(self, outputs_dim, d_model=32):
        super(RlNet, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(10 * d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, outputs_dim)
                    )
    def forward(self, extract):
        act = self.net(extract.transpose(1, 0).reshape(-1, 320))
        return act

class TsNet(nn.Module):
    def __init__(self, N_STATES, outputs_dim):
        super(TsNet, self).__init__()
        self.encoder = TransAm(N_STATES, outputs_dim)
        
    def forward(self, s):
        extract = self.encoder(s)
        return extract

class ArNet(nn.Module):
    def __init__(self, N_STATES, d_model=32):
        super(ArNet, self).__init__()
        self.net = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, N_STATES)
                        )
    def forward(self, extract):
        return self.net(extract)

class TS():
    def __init__(self):
        self.TS_estimate = TsNet(N_STATES, outputs_dim).to(device)
        self.TS_target = TsNet(N_STATES, outputs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.TS_estimate.parameters(), lr = LR_TS)

    def encode(self, s):
        s_extract = self.TS_estimate(s)
        return s_extract
    
    def encode_target(self, s):
        s_extract_target = self.TS_target(s)
        return s_extract_target
    
    def learn(self, loss_Q):
        loss = loss_Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.TS_target.parameters(), 
                                    self.TS_estimate.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class RL():
    def __init__(self):
        self.RL_estimate = RlNet(outputs_dim).to(device)
        self.RL_target = RlNet(outputs_dim).to(device)
        self.optimizer = torch.optim.Adam(self.RL_estimate.parameters(), lr = LR_RL)
        self.lossfun = nn.MSELoss()

    def select_action(self, extract):
        return self.RL_estimate(extract).detach()

    def loss_for_ts(self, extract):
        Q_estimate = -1 * self.RL_estimate(extract).min(1)[0].mean()
        return Q_estimate

    def learn(self, extract, a, r, s_, mask):
        Q_estimate = self.RL_estimate(extract).gather(1, a)
        Q_next = torch.zeros(extract.shape[1], device=device)
        Q_next[mask] = self.RL_target(s_).max(1)[0].detach()
        #Q_next[mask] = self.RL_target(s_).mean(dim = 1).detach()
        Q_target = r + GAMMA * Q_next
        loss = self.lossfun(Q_estimate, Q_target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
    
    def soft_update(self):
        for target_param, param in zip(self.RL_target.parameters(),
                                       self.RL_estimate.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


############################################ Training #################################

model_ts = TS()
model_rl = RL()

memory = ReplayMemory(1000000)
steps_done = 0

explore = 0
total_decis = 0

def select_action(state):
    global steps_done, total_decis, explore
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    total_decis += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #return policy_net(state)['rl'].max(1)[1].view(1, 1)
            extract = model_ts.encode(state)
            return model_rl.select_action(extract).max(1)[1].view(1, 1)
            #return model_rl(model_ts(state)).max(1)[1].view(1, 1)
    else:
        explore += 1
        return torch.tensor([[random.randrange(outputs_dim)]], device=device, dtype=torch.long)

def valid_plot(true, pred, episode):
    x_true = true[-1, :].to("cpu").detach().numpy() 
    x_pred = pred[-1, :].to("cpu").detach().numpy() 
    for i in range(4):
        plt.plot(x_true[:, i], color="red", label='true')
        plt.plot(x_pred[:, i], color="blue", label='pred')
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        plt.legend()
        plt.savefig('graph/transformer-episode{}-{}.png'.format(episode, i))
        plt.close()

############################################ Training loop #################################
ad_count = 0
total_adLoss = []
total_RLoss = []

def optimize_model():
    global i_episode, ad_count, update_ad
    if len(memory) < BATCH_SIZE:
        return
    update_ad += 1
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None], dim=1)
    state_batch = torch.cat(batch.state, dim=1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_states = model_ts.encode(non_final_next_states).detach()
    extract = model_ts.encode(state_batch)
    model_rl.learn(extract, action_batch, reward_batch, next_states, non_final_mask)
    loss_ts = model_rl.loss_for_ts(extract)
    model_ts.learn(loss_ts)
    model_ts.soft_update()
    model_rl.soft_update()
    
    

num_episodes = 2000
const_reward_total = 0
sequenceState = SequenceStates(max_length)
update_ad = 0

reward_max = 0
try:
    with open(r'./cartpole.log', 'r') as f:
        fd = f.readlines()
    reward_max = float(re.findall(r'\d+', fd[-1])[0])
    model_ts.load_state_dict(torch.load(r'./cartpoleBestTs.pth'))
    model_rl.load_state_dict(torch.load(r'./cartpoleBestRl.pth'))
    print("加载模型>>>>reward={}".format(reward_max))
except Exception as e:
    print(e)
for i_episode in range(num_episodes):
    # Initialize the environment and state
    reward_total = 0
    state = env.reset()
    state = torch.from_numpy(state).to(device)
    for i in range(max_length):
        sequenceState.push(state)
    state = sequenceState.get()
    for t in count():
        # Select and perform an action
        action = select_action(state)
        s, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        with torch.no_grad():
            reward_total += reward.cpu().item()
        # Observe new state
        s = torch.from_numpy(s).to(device)
        if not done:
            sequenceState.push(s)
            next_state = sequenceState.get()
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            break
    if i_episode % 1 == 0:
        const_reward_total = reward_total
        ad_count = 0
        print("Episode {} | Total reward is {} ".format(i_episode ,reward_total))
        print('AD loss: {}. RL loss: {}'.format(np.mean(total_adLoss), np.mean(total_RLoss)))
        total_RLoss = []
        total_adLoss = []
    # Update the target network, copying all weights and biases in DQN
    if reward_total > reward_max:
        reward_max = reward_total
        with open(r'./cartpole.log', 'a') as f:
            f.write(str(reward_total) + '\n')
        torch.save(model_ts.state_dict(), r'./cartpoleBestTs.pth')
        torch.save(model_rl.state_dict(), r'./cartpoleBestRl.pth')

print('Complete')
env.render()
env.close()
