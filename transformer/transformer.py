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
import re


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
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #单词在词汇表中的位置？
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return x + self.pe[:x.size(0), :]

#feature_size:词特征向量的维度
class TransAm(nn.Module):
    def __init__(self, feature_size=64, num_layers=3, dropout=0.5):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
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
        #30x64x1
        src = self.pos_encoder(src)
        #30x64x250
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        #30x64x1 seq2seq
        
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
    #series = pd.read_csv('./data/1863690462_1_9260_.csv').iloc[:, 5]
    series = pd.read_csv(
    '../capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/resources/workload/sample/sample_no_anomaly.csv'
    ).iloc[:, 1]
    global resource_type 
    resource_type = series.name
    if resource_type != load_model.split('/')[2]:
        print('ERROR: The resource specific is {}. But found model load is {}'.format(resource_type, load_model.split('/')[2]))
        sys.exit(0)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()
    series = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    
    train_samples = int(0.7 * len(series))
    train_data = series[:train_samples]
    test_data = series[train_samples:]

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]

    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]

    return train_sequence.to(device), test_data.to(device)

def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))

    return input, target

def train(train_data):
    model.train()

    for batch_index, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        start_time = time.time()
        total_loss = 0
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        #log_interval = int(len(train_data) / batch_size / 5)
        log_interval = plot_freq
        #if batch_index % log_interval == 0 and batch_index > 0:
        if epoch % log_interval == 0 and batch_index > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.9f} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'
                  .format(epoch, batch_index, len(train_data) // batch_size, scheduler.get_lr()[0], elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))

def evaluate(eval_model, data_source):
    eval_model.eval()
    total_loss = 0
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)
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
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
    for i in range(plot_spans):
        bins = len(test_result) // plot_spans
        plt.plot(test_result[i*bins: (i + 1) * bins], color="red")
        plt.plot(truth[i*bins: (i + 1) * bins], color="blue")
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        try:
            os.makedirs('graph-%s/epoch-%d' % (resource_type, epoch))
        except Exception as e:
            pass
        plt.savefig('graph-%s/epoch-%d/%d.png' % (resource_type, epoch, i))
        plt.close()

    return total_loss / len(test_result)

def readMinLoss(file):
    print('#INFO load loss record from ' + file)
    with open(file, 'r') as f:
        loss = f.read()
    loss = re.findall('\d+.\d+', loss)[0]
    print('min loss is ' + loss)
    return float(loss)
load = False
load_model = r'./pkl/mean cpu usage/transformer.pt'
train_data, val_data = get_data()
model = TransAm().to(device)
criterion = nn.MSELoss()
#lr = 0.005
lr = 1e-9
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=1)
#gamma=0.95
#gamma=0.99 lr=0.005 step=1
epochs = 1000  # The number of epochs

plot_freq = 20
plot_spans = 20# 原数据太长，分成plot_spans段分别画出来
valid_loss_min = readMinLoss('./pkl/mean cpu usage/loss.txt')
model_save = r'./pkl'
global resource_type

if load:
    try:
        model_state_dict = torch.load(load_model)
        model.load_state_dict(model_state_dict)
        print('#INFO model loaded')
    except Exception as e:
        print('#INFO Not Found model...')

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)

    if (epoch % plot_freq is 0):
        val_loss = plot_and_loss(model, val_data, epoch)
    else:
        val_loss = evaluate(model, val_data)
    if epoch % plot_freq == 0:
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (
                    time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
    scheduler.step()
    if valid_loss_min > val_loss:
        valid_loss_min = val_loss
        savePth = model_save + '/' + resource_type
        try:
            os.makedirs(savePth)
        except Exception as e:
            pass
        torch.save(model.state_dict(), savePth + '/transformer.pt')
        print('#INFO model saved loss is ' + str(val_loss))
        with open(savePth + '/loss.txt', 'w') as f:
            f.write('loss_min: {}'.format(valid_loss_min))