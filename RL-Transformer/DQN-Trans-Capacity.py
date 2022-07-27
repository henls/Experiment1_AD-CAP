from cProfile import label
import itertools
import re
from sys import prefix
import time
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
anomaly_side = 0
anomaly_error = []
util_max = 0.8
util_min = 0.2
duel = 0
if duel:
    print("使用duel DQN")
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

class Net(nn.Module):
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.):
        super(Net, self).__init__()
        self.encoder = TransAm(N_STATES, outputs)
        self.d_model = d_model
        self.out = nn.Sequential(
                        nn.Linear(d_model, outputs)
                        )
        self.DqnOut_metric = nn.Sequential(
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
        #transformer应该只对序列编码，RT等特征需要单独提取
        #x:sequence x batchsize x dim
        x = x.float()
        #x_metric = self.DqnOut_metric((x[12] - x[13]).transpose(1,0).reshape(-1, 1))
        x_metric = self.DqnOut_metric((x[[9, 12, 13]]).transpose(1,0).reshape(-1, 3))
        # 响应时间、当前虚拟机已用资源、当前虚拟机总容量、任务数
        #x = torch.cat([x_usage_rl, x_metric]).reshape(x.shape[1], -1)#两个特征融合后送到dqnout决策。
        if duel:
            ad = self.ad(x_metric)
            val = self.val(x_metric).expand(x_metric.size(0), ad.size(1))
            rlout = val + ad - ad.mean(1).unsqueeze(1).expand(x_metric.size(0), ad.size(1))
        else:
            rlout = self.out(x_metric)
        #batch x d_model
        return {'rl':rlout}

class AdNet(nn.Module):
    def __init__(self, N_STATES, outputs, d_model=32):
        super(AdNet, self).__init__()
        self.encoder = TransAm(N_STATES, outputs)
        self.d_model = d_model
        self.AdOut = nn.Sequential(
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, d_model),
                        nn.ReLU(),
                        nn.Linear(d_model, N_STATES)
                        )
    def forward(self, x):
        x = x.float()
        x_usage = self.encoder(x[:10])#得到序列特征
        adout = self.AdOut(x_usage)#时间窗口内的数据（利用率数据）
        return {'ad':adout}

############################################ Training #################################
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05#0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

screen_height, screen_width = 1, 4

# Get number of actions from gym action space
n_actions = env.actionSpace()
n_status = env.statusSpace()
policy_net = Net(n_status, n_actions).to(device)
target_net = Net(n_status, n_actions).to(device)
ad_net = AdNet(n_status, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters())
#optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)

optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)
optimizer_AD = optim.Adam(ad_net.parameters(), lr = 1e-4)
memory = ReplayMemory(1000000)
memory_ad = ReplayMemory(1000000)
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

def isAnomaly(s, s_):
    
    if len(anomaly_error) < 1000:
        return False
    s_pred = ad_net(s_)['ad'][-1]
    if s_pred > util_max or s_pred < util_min:
        return True
    cat_err = torch.cat(anomaly_error, dim = 0)
    mu, sigma = torch.mean(cat_err), torch.std(cat_err)
    s_last_pred = ad_net(s)['ad'][-1]
    loss = s_[-1] - s_last_pred
    if loss > mu + 2 * sigma or loss < mu - 2 * sigma:
        return True
    else:
        return False

    

############################################ Training loop #################################

ad_count = 0
total_adLoss = []
total_RLoss = []
DDQN = True
def optimize_ad():
    global i_episode, ad_count, total_adLoss, total_RLoss
    if len(memory_ad) < BATCH_SIZE:
        return 
    transitions = memory_ad.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None], dim=1).to(torch.float32)
    state_batch = torch.cat(batch.state, dim=1)
    criterion = nn.MSELoss()
    pred = ad_net(state_batch[:, non_final_mask])['ad']
    with torch.no_grad():
        if i_episode % 1 == 0 and ad_count == 0:
            ad_count = 1
            #验证网络预测能力
            valid_plot(non_final_next_states[:10, :], pred, i_episode)
        err = non_final_next_states[:10, :] - pred
        for i in range(len(pred)):
            anomaly_error.append(err[i])
    loss_ad = criterion(pred, non_final_next_states[:10, :])
    loss_AD = torch.log(1+loss_ad)
    total_adLoss.append(np.log(1+loss_AD.item()))
    optimizer_AD.zero_grad()
    loss_AD.backward()
    optimizer_AD.step()

def optimize_rl():
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

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #DDQN
    if (DDQN):
        with torch.no_grad():
            q_tp1_values = policy_net(non_final_next_states)['rl'].detach()
            _, a_prime = q_tp1_values.max(1)
            q_s_a = target_net(non_final_next_states)['rl'].detach()
            next_state_values[non_final_mask] = q_s_a.gather(1, a_prime.unsqueeze(1)).squeeze()
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states)['rl'].max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    clipped_loss = loss.clamp(-1, 1)
    # loss_merge = torch.log(1+loss) + torch.log(1+loss_ad)
    # loss_merge = torch.log(1+loss)
    loss_merge = loss
    # Optimize the model
    optimizer.zero_grad()
    loss_merge.backward()
    # clipped_loss.backward()
    total_RLoss.append(np.log(1+loss.item()))
    '''for param in policy_net.parameters():
        try:
            param.grad.data.clamp_(-1, 1)
        except Exception as e:
            pass'''
    optimizer.step()
num_episodes = 2000
sequenceState = SequenceStates(max_length)
reward_max = -1e5
if duel:
    log_pth = r'./save-capacity-duel.log'
    rl_pth = r'./BestModel-capacity-duel-rl.pth'
    ad_pth = r'./BestModel-capacity-duel-ad.pth'
else:
    log_pth = r'./save-capacity.log'
    rl_pth = r'./BestModel-capacity-rl.pth'
    ad_pth = r'./BestModel-capacity-ad.pth'
try:
    with open(log_pth, 'r') as f:
        fd = f.readlines()
    reward_max = float(re.findall(r'\d+', fd[-1])[0])
    policy_net.load_state_dict(torch.load(rl_pth))
    ad_net.load_state_dict(torch.load(ad_pth))
    print("加载模型>>>>reward={}".format(reward_max))
except Exception as e:
    print(e)
anomaly = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    reward_total = 0
    state = env.reset()
    #13 x 1.长度13，维度1
    state = torch.from_numpy(state).reshape(max_length, -1, n_status).to(device)
    last_state = state
    for t in count():
        # Select and perform an action
        anomaly = isAnomaly(last_state, state)
        if anomaly:
            action = select_action(state)
        else:
            action = torch.IntTensor([0])
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
        if anomaly:
            memory.push(state, action, next_state, reward)
        memory_ad.push(state, action, next_state, reward)
        # Move to the next state
        last_state = state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_rl()
        optimize_ad()
        if done:
            break
    if i_episode % 1 == 0:
        ad_count = 0
        print("Episode {} | Total reward is {} ".format(i_episode ,reward_total))
        print('AD loss: {}. RL loss: {}'.format(np.mean(total_adLoss), np.mean(total_RLoss)))
        total_RLoss = []
        total_adLoss = []
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if reward_total > reward_max and i_episode > 1:
        with open(log_pth, 'a') as f:
            f.write(str(reward_total) + '\n')
        reward_max = reward_total
        torch.save(policy_net.state_dict(), rl_pth)
        torch.save(ad_net.state_dict(), ad_pth)

print('Complete')
env.render()
env.close()
