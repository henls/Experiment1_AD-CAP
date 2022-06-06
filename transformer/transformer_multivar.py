from cProfile import label
from email.utils import collapse_rfc2231_value
from turtle import color
import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import sys

#多维回归版
os.environ["CUDA_VISIBLE_DEVICES"]='1'

torch.manual_seed(0)
np.random.seed(0)

input_window = 30
output_window = 1
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        #pe = torch.zeros(max_len, d_model, 2)#2是数据维度
        pe = torch.zeros(max_len, d_model)#2是数据维度
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #单词在词汇表中的位置？
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.trans_multivar_layer1 = nn.Linear(2, d_model)
        self.DropOut = nn.Dropout(p=0.1)
        #self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        #nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        
        X = self.trans_multivar_layer1(x).squeeze(axis=2) + self.pe[:x.size(0), :]
        return self.DropOut(X)

#feature_size:词特征向量的维度
class TransAm(nn.Module):
    def __init__(self, feature_size=64, num_layers=6, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)

        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 2)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)

        
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)

def get_data():

    #series = pd.read_csv('./000001_Daily.csv', usecols=['Close'])
    #series = pd.read_csv('./data/1863690462_1_9260_.csv').iloc[:, [4, 5]]
    series = pd.read_csv(
    '../capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/resources/workload/sample/sample.csv'
    ).iloc[:, [1,2]]
    global resource_type 
    resource_type = series.columns
    global load_model
    check_resource_type(resource_type, load_model)
    resource_type = '_'.join(resource_type)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    
    if OnlineBN == False:
        scaler = StandardScaler()
        
        series = scaler.fit_transform(series.values.reshape(-1, series.shape[-1]))
    else:
        series = series.values.reshape(-1, series.shape[-1])
    
    train_samples = int(0.7 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]

    train_sequence = create_inout_sequences(train_data, input_window)
    
    train_sequence = train_sequence[:-output_window]
    
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]
    return train_sequence.to(device), test_data.to(device)

def check_resource_type(resource_type, load_model):
    resource = load_model.split('/')[2].split('_')
    type_num = 0
    for i in resource:
        if i in '_'.join(resource_type):
            type_num += 1
        
    if type_num != len(resource):
        print('ERROR: The resource specific is {}. But found model load is {}'.format(resource_type, load_model.split('/')[2]))
        sys.exit(0)

def get_batch(source, i, batch_size):
    alpha = 0.5
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    global mean, std
    if OnlineBN:
        batch_mean, batch_std = torch.mean(data, axis=(0, 1)), torch.std(data, axis=(0,1))
        if i == 0:
            mean, std = batch_mean, batch_std
        else:
            mean = alpha * mean + (1 - alpha) * batch_mean
            std = alpha * std + (1 - alpha) * batch_std
        data = (data - mean) / (std + 1e-10)
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 沿1轴分成input window块
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target

def train(train_data):
    model.train()
    
    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size)
        
        targets = targets.squeeze()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item() * len(data[0])
        
        log_interval = plot_freq
        
        if epoch % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / batch_index
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
    train_loss.append(total_loss / len(train_data))
    #print('train_loss', train_loss[-1])
def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
            targets = targets.squeeze(axis=-2)
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i, 1)
            
            output = eval_model(data)
            target = target.squeeze(axis=-2)
            
            total_loss += len(data)*criterion(output, target).cpu().item()
            
            test_result = torch.cat((test_result, output[-1].cpu()), 0)
            truth = torch.cat((truth, target[-1].cpu()), 0)

    for i in range(plot_spans):
        bins = len(test_result) // plot_spans
        for idx, resource in enumerate(resource_type.split('_')):
            plt.plot(test_result[i*bins: (i + 1) * bins, idx], color="red", label = 'pred')
            plt.plot(truth[i*bins: (i + 1) * bins, idx], color="blue", label = 'truth')
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            try:
                os.makedirs('graph-%s/epoch-%d' % (resource_type, epoch))
            except Exception as e:
                pass
            plt.savefig('graph-%s/epoch-%d/%d-%s.png' % (resource_type, epoch, i, resource))
            plt.close()
    
    return total_loss / len(data_source)
global OnlineBN
OnlineBN = False
global mean, std
mean, std = 0, 0

load_model = r'./pkl/cpu_mem/transformer.pt'
train_data, val_data = get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
#lr = 0.005
lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.95)
#gamma=0.95
#gamma=0.99 lr=0.005 step=1
epochs = 100  # The number of epochs

plot_freq = 1
plot_spans = 20# 原数据太长，分成plot_spans段分别画出来
valid_loss_min = 1e10
model_save = r'./pkl'
global resource_type
load_model = r'./pkl/mean cpu usage_canonical memory usage/transformer.pt'


try:
    model_state_dict = torch.load(load_model)
    model.load_state_dict(model_state_dict)
    print('INFO# Loaded model...')
except Exception as e:
    print('INFO# Not Found model...')
valid_loss = []
train_loss = []

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % plot_freq is 0):
    #if 0:
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)
    valid_loss.append(val_loss)
    if epoch % plot_freq == 0:
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                    time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        #print('val_loss', val_loss)
    scheduler.step()
    if valid_loss_min > val_loss:
        valid_loss_min = val_loss
        savePth = model_save + '/' + resource_type
        try:
            os.makedirs(savePth)
        except Exception as e:
            pass
        torch.save(model.state_dict(), savePth + '/transformer.pt')
        with open(savePth + '/loss.txt', 'w') as f:
            f.write('loss_min: {}'.format(valid_loss_min))

plt.plot(train_loss, color='red', label='train')
plt.plot(valid_loss, color='blue', label='test')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.savefig('graph-loss/cpu_mem.png')
plt.close()