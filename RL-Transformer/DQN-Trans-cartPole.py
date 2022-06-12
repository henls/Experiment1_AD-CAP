#使用图片作为状态空间
from cProfile import label
import itertools
from sys import prefix
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


from line_profiler import LineProfiler
############################################ init #################################
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
env = gym.make('CartPole-v0').unwrapped
env.seed(1000)
torch.manual_seed(200)
random.seed(60)
max_length = 10
dqn_from_ad = False
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

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
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.):
        super(TransAm, self).__init__()
        global max_length
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=1, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.decoder = nn.Linear(feature_size, 2)
        #self.init_weights()
        self.fc1 = nn.Linear(N_STATES,16) # 4->16
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(16, d_model) # 16 -> 32
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(d_model, d_model) # 32 -> 32
        self.fc3.weight.data.normal_(0, 0.1)
        self.out1 = nn.Linear(d_model, outputs)
        self.out1.weight.data.normal_(0, 0.1)


    '''def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)'''

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(max_length).to(device)
            self.src_mask = mask
        src = F.relu(self.fc1(src))
        src = F.relu(self.fc2(src))
        src = F.relu(self.fc3(src))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #10x128x32
        #output = self.out1(output[-1, :, :])#验证cartpole问题
        #output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class DQN(nn.Module):
    def __init__(self, N_STATES, outputs, ad_model, d_model=32, num_layers=1, dropout=0.):
        super(DQN, self).__init__()
        self.encoder = TransAm(N_STATES, outputs)
        self.encoder.load_state_dict(ad_model.encoder.state_dict())
        #self.out1 = nn.Linear(d_model, 2*d_model)
        self.out1 = nn.Linear(d_model, outputs)
        self.out1.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(2 * d_model, 2 * d_model)
        self.out2.weight.data.normal_(0, 0.1)
        self.out3 = nn.Linear(2 * d_model, d_model)
        self.out3.weight.data.normal_(0, 0.1)
        self.out4 = nn.Linear(d_model, outputs)
        self.out4.weight.data.normal_(0, 0.1)


    def forward(self, x):
        if dqn_from_ad:
            x = self.encoder(x).detach()
        else:
            x = self.encoder(x)
        '''x = F.relu(self.out1(x[-1, :]))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        output = self.out4(x)'''
        output = self.out1(x[-1, :])
        return output

class AD(nn.Module):
    def __init__(self, N_STATES, outputs, d_model=32, num_layers=1, dropout=0.5):
        super(AD, self).__init__()
        self.encoder = TransAm(N_STATES, outputs, dropout=dropout)
        #self.out = nn.Linear(d_model, N_STATES)
        #self.out.weight.data.normal_(0, 0.1)
        self.out1 = nn.Linear(d_model, 2 * d_model)
        self.out1.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(2 * d_model, 4 * d_model)
        self.out2.weight.data.normal_(0, 0.1)
        self.out3 = nn.Linear(4 * d_model, 8 * d_model)
        self.out3.weight.data.normal_(0, 0.1)
        self.out4 = nn.Linear(8 * d_model, 8 * d_model)
        self.out4.weight.data.normal_(0, 0.1)
        self.out5 = nn.Linear(8 * d_model, 4 * d_model)
        self.out5.weight.data.normal_(0, 0.1)
        self.out6 = nn.Linear(4 * d_model, 2 * d_model)
        self.out6.weight.data.normal_(0, 0.1)
        self.out7 = nn.Linear(2 * d_model, N_STATES)
        self.out7.weight.data.normal_(0, 0.1)
        
    
    def forward(self, x):
        if dqn_from_ad:
            x = self.encoder(x)
        else:
            x = self.encoder(x).detach()
        #output = self.out(x)
        x = F.relu(self.out1(x))
        x = F.relu(self.out2(x))
        x = F.relu(self.out3(x))
        x = F.relu(self.out4(x))
        x = F.relu(self.out5(x))
        x = F.relu(self.out6(x))
        output = self.out7(x)
        return output

############################################ Input extraction #################################

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


############################################ Training #################################
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

screen_height, screen_width = 1, 4

# Get number of actions from gym action space
n_actions = env.action_space.n
anomaly_net = AD(4, n_actions).to(device)
policy_net = DQN(4, n_actions, anomaly_net).to(device)
target_net = DQN(4, n_actions, anomaly_net).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(), lr = 1e-3)


optimizer_ad = optim.Adam(anomaly_net.parameters(), lr = 1e-6)
scheduler_ad = LambdaLR(optimizer_ad, lr_lambda=lambda t: 0.99**(t/100))

optimizer_merge = optim.Adam([
    {'params': policy_net.parameters(), 'lr': 1e-3}, 
    {'params': anomaly_net.parameters()}
    ])

memory = ReplayMemory(10000)
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        explore += 1
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

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

episode_durations = []

############################################ Training loop #################################
ad_count = 0
def optimize_model():
    global i_episode, ad_count
    state_valid_pred = []
    state_valid_true = []
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None], dim=1)
    state_batch = torch.cat(batch.state, dim=1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    pred = anomaly_net(state_batch[:, non_final_mask])
    loss_ad = criterion(pred, non_final_next_states)

    loss_merge = loss + loss_ad
    # Optimize the model
    if i_episode <= 100:
        optimizer.zero_grad()
    optimizer_ad.zero_grad()
    #optimizer_merge.zero_grad()
    if i_episode <= 100:
        loss.backward()
    loss_ad.backward()
    #loss_merge.backward()
    if i_episode % 50 == 0 and ad_count == 0:
        print('AD loss: ', loss_ad.item(), 'lr ', optimizer_ad.param_groups[0]['lr'])
        ad_count = 1
        #验证网络预测能力
        valid_plot(non_final_next_states, pred, i_episode)
    for param in policy_net.parameters():
        try:
            param.grad.data.clamp_(-1, 1)
        except Exception as e:
            pass
    '''for param in anomaly_net.parameters():
        try:
            param.grad.data.clamp_(-1, 1)
        except Exception as e:
            pass'''
    if i_episode <= 100:
        optimizer.step()
    optimizer_ad.step()
    #scheduler_ad.step()
    #optimizer_merge.step()
num_episodes = 2000
sequenceState = SequenceStates(max_length)
for i_episode in range(num_episodes):
    # Initialize the environment and state
    reward_total = 0
    state = env.reset()
    state = torch.from_numpy(state).to(device)
    for i in range(max_length):
        sequenceState.push(state)
    state = sequenceState.get()
    #matplotlib.image.imsave('./state.jpg', np.array(state[0, 0,:,:]))
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
            episode_durations.append(t + 1)
            #plot_durations()
            break
    if i_episode % 50 == 0:
        ad_count = 0
        print("Episode {} | Total reward is {} ".format(i_episode ,reward_total))
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()