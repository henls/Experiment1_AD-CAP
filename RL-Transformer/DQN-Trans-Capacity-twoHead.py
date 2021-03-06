from cProfile import label
import itertools
import re
from sys import prefix
import time
from turtle import forward
from xml.dom.pulldom import parseString
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR

import json

from line_profiler import LineProfiler
############################################ init #################################
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
torch.manual_seed(200)
random.seed(60)
max_length = 14
dqn_from_ad = False
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
############################################ replay memory #################################
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class envCapacity(object):

    def __init__(self, n_action, n_status, config):
        self.n_action  = n_action
        self.n_status = n_status
        self.config = config
        self.loadconfig()


    def loadconfig(self):
        with open(self.config, 'r') as fd:
            content = json.load(fd)
        self.action_path = content['action']
        self.status_path = content['status']
        self.max_length = content['maxlength']
    
    def actionSpace(self):
        return self.n_action
    
    def statusSpace(self):
        return self.n_status

    def reset(self):
        ini = [0.]*self.max_length
        return np.array(ini, dtype=np.float32).reshape(self.max_length, -1)
    
    def step(self, action):
        #返回长度为14，维度为1的序列。cartpole任务返回长度10，维度4的序列
        fd = 'null'
        action -= 1
        while 'null' in fd:
            assert action in [-1, 0, 1]
            fd = ''
            while 'OK' not in fd:
                with open(self.action_path, 'r') as f:
                    fd = f.read() 
                time.sleep(0.1)
            with open(self.action_path, 'w') as f:
                f.write(str(action)) 
            fd = 'OK'
            while 'OK' in fd:
                with open(self.status_path, 'r') as f:
                    fd = f.read()
            with open(self.status_path, 'w') as f:
                f.write(fd + 'OK')
        p, reward, done = fd.split('&')[0], float(fd.split('&')[1]), float(fd.split('&')[2])
        cpu, RT, usedPEs, totalpes, cloudletNums = p.split('$')
        cpu = [float(i) for i in cpu[1:-1].split(',')]
        RT = float(RT)
        usedPEs = float(usedPEs)
        totalpes = float(totalpes)
        cloudletNums = float(cloudletNums)
        return np.array(cpu + [RT, usedPEs, totalpes, cloudletNums]).reshape(self.max_length, -1), reward, done

env = envCapacity(3, 1, 'config/configure')

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
        #capacity任务中不需要这个也可以，因为每次追加的长度就是管道长度，相当于替换了
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
                    nn.ReLU(),
                )

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(10).to(device)
            self.src_mask = mask
        src = src.to(torch.float32)
        src = self.fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)#特征提取网络
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class AD(nn.Module):
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.):
        super(AD, self).__init__()
        self.encoder = TransAm(N_STATES, outputs)
        self.AdOut = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, N_STATES)
                        )
    def forward(self, x):
        #transformer应该只对序列编码，RT等特征需要单独提取
        #x:sequence x batchsize x dim
        x = x.float()
        x_usage = self.encoder(x[:10])#得到序列特征
        adout = self.AdOut(x_usage)#时间窗口内的数据（利用率数据）
        #batch x d_model
        return {'ad':adout}

class RL(nn.Module):
    #非序列版
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.):
        super(RL, self).__init__()
        self.extract = nn.Sequential(
                        nn.Linear(3, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU()
                        )
        self.ad = nn.Sequential(
                        nn.Linear(d_model, 16),
                        nn.ReLU(),
                        nn.Linear(16, outputs)
                        ) 
        self.val = nn.Sequential(
                        nn.Linear(d_model, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                        )
    def forward(self, x):
        #x:sequence x batchsize x dim
        x = x.float()
        x[12] = 32 - x[12]
        feature_extract = self.extract(x[[9,12,13]].transpose(1,0).reshape(-1, 3))#当前时刻利用率 + RT + 总可用核数 + 任务数
        ad = self.ad(feature_extract)
        val = self.val(feature_extract).expand(feature_extract.size(0), ad.size(1))
        rlout = val + ad - ad.mean(1).unsqueeze(1).expand(feature_extract.size(0), ad.size(1))
        #batch x d_model
        return {'rl':rlout}


############################################ Input extraction #################################


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

############################################ Training #################################
#BATCH_SIZE = 128
#GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05#0.05
EPS_DECAY = 6000
TARGET_UPDATE = 5

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

screen_height, screen_width = 1, 4

# Get number of actions from gym action space
n_actions = env.actionSpace()
n_status = env.statusSpace()
policy_net = RL(n_status, n_actions).to(device)
target_net = RL(n_status, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters())
#optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)
optimizer_rl = optim.Adam(policy_net.parameters(), lr = 1e-3, weight_decay=0.001)
#optimizer_rl = optim.RMSprop(policy_net.parameters(), alpha=0.95, eps=0.01, weight_decay=0.001)
scheduler_1 = LambdaLR(optimizer_rl, lr_lambda=lambda epoch: np.exp(-epoch/6000))
#optimizer_ad = optim.Adam([{"params": policy_net.encoder.parameters()}, {"params": policy_net.AdOut.parameters()}], lr = 1e-3)
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
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state)['rl'].max(1)[1].view(1, 1)
    else:
        explore += 1
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def valid_plot(true, pred, episode):
    x_true = true[-1, :].to("cpu").detach().numpy() 
    x_pred = pred[-1, :].to("cpu").detach().numpy() 
    plt.plot(x_true, color="red", label='true')
    plt.plot(x_pred, color="blue", label='pred')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.legend()
    plt.savefig('graph/transformer-episode{}.png'.format(episode))
    plt.close()

############################################ Training loop #################################
ad_count = 0
total_adLoss = []
total_RLoss = []
def optimize_model():
    global i_episode, ad_count, total_adLoss, total_RLoss
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None], dim=1).to(torch.float32)
    state_batch = torch.cat(batch.state, dim=1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch)['rl'].gather(1, action_batch)
    #Double DQN
    q_tp1_values = policy_net(non_final_next_states)['rl'].detach()
    _, a_prime = q_tp1_values.max(1)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[non_final_mask] = target_net(non_final_next_states)['rl'].max(1)[0].detach()
    q_s_a = target_net(non_final_next_states)['rl'].detach()
    next_state_values[non_final_mask] = q_s_a.gather(1, a_prime.unsqueeze(0))
    #a = target_net(non_final_next_states)['rl']
    #next_state_values[non_final_mask] = torch.gather(a, 1, torch.randint(3, (1, a.shape[0]), device=device)).detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer_rl.zero_grad()
    loss.backward()
    total_RLoss.append(np.log(1+loss.item()))
    '''if i_episode % 1 == 0 and ad_count == 0:
        ad_count = 1
        #验证网络预测能力
        valid_plot(non_final_next_states[:10, :], pred, i_episode)'''
    for param in policy_net.parameters():
        try:
            param.grad.data.clamp_(-1, 1)
        except Exception as e:
            pass
    optimizer_rl.step()
    scheduler_1.step()
num_episodes = 100
sequenceState = SequenceStates(max_length)
#BATCH_SIZE = 64
#GAMMA = 0.98
reward_max = -1e5
try:
    with open(r'./save.log', 'r') as f:
        fd = f.readlines()
    reward_max = float(re.findall(r'[-]\d+', fd[-1])[0])
    policy_net.load_state_dict(torch.load(r'./BestModel.pth'))
    print("加载模型>>>>reward={}".format(reward_max))
except Exception as e:
    print(e)
for BATCH_SIZE in range(32, 256, 32):
    for GAMMA_s in range(999, 850, -5):
        GAMMA = GAMMA_s / 1000
        total_adLoss = []
        total_RLoss = []
        reward_mean = []
        GAMMA = 0.9
        BATCH_SIZE = 32
        for i_episode in range(num_episodes):
            # Initialize the environment and state
            reward_total = 0
            state = env.reset()
            #13 x 1.长度13，维度1
            state = torch.from_numpy(state).reshape(max_length, -1, n_status).to(device)
            for t in count():#功能等价while，但是可以自动计循环次数
                # Select and perform an action
                action = select_action(state)
                s, reward, done = env.step(action.item())
                #状态归一化
                #s[11:] /= 32
                reward = torch.tensor([reward], device=device)
                with torch.no_grad():
                    reward_total += reward.cpu().item()
                # Observe new state
                s = torch.from_numpy(s).reshape(max_length, -1, n_status).to(device)
                if not done:
                    next_state = s
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
                ad_count = 0
                print("Episode {} | Total reward is {} ".format(i_episode ,reward_total))
                print('AD loss: {}. RL loss: {}. LR: {}'.format(np.mean(total_adLoss), np.mean(total_RLoss), optimizer_rl.param_groups[0]['lr']))
                total_RLoss = []
                total_adLoss = []
                '''params = policy_net.parameters()
                k = 0
                for idx, i in enumerate(params):
                    l = 1
                    print("第 {} 层参数和 {} 均值 {}".format(idx, torch.sum(i).item(), torch.mean(i).item()))
                    for j in i.size():
                        l *= j
                    k = k + l'''
            # Update the target network, copying all weights and biases in DQN
            reward_mean.append(reward_total)
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if reward_total > reward_max:
                with open(r'./save.log', 'a') as f:
                    f.write(str(reward_total) + '\n')
                reward_max = reward_total
                torch.save(policy_net.state_dict(), r'./BestModel.pth')
        """reward_mean.remove(min(reward_mean))
        reward_mean.remove(max(reward_mean))
        total_RLoss.remove(min(total_RLoss))
        total_RLoss.remove(max(total_RLoss))
        print('Toatal reward: {}. RL loss: {}. GAMMA: {}. BATCH: {}'.format(np.mean(reward_mean), 
                                                                np.mean(total_RLoss), 
                                                                GAMMA,
                                                                BATCH_SIZE
                                                                ))"""
print('Complete')
env.render()
env.close()
