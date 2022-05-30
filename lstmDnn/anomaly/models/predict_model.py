#import tensorflow as tf
from curses import raw
from email import header
from sklearn.metrics import r2_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import json
import sys
import os
import datetime
import pandas as pd
sys.path.append(r'/home/wxh/lstmDnn/anomaly')
sys.path.append(r'/home/wxh/lstmDnn/anomaly/safekit')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#from safekit.batch import OnlineBatcher
from safekit.graph_training_utils import ModelRunner, EarlyStop
#from safekit.tf_ops import lm_rnn
import safekit.tf_ops as ops
from safekit.util import get_mask, Parser

import customTools
import customTools.utils as cutils


class model():

    def __init__(self):
        with open(r'/home/wxh/lstmDnn/anomaly/pkl/save_normal.log', 'r') as f:
            a = f.readlines()
        
        #self.loss = 0.00056153577
        self.loss = float(a[-1].strip().split(' ')[-1])
        self.pointless = []
        self.earlystop_count = 10

    def saveObserved(self,pth, idx):
        fname = os.path.dirname(pth) + '/observed_' + os.path.basename(pth)
        [idx_st, idx_ed] = idx
        if os.path.exists(fname) == 0:
            df = pd.read_csv(pth, header=None)
            df = df.iloc[idx_st: idx_ed,:-3]
            #columns=["ST", "ET", "jobID", "taskID", "machineID", "mean cpu usage"," canonical memory usage", 
            #    "unmapped page cache", "total page cache", "disk io", "CPI", "MAI"]
            
            df = pd.DataFrame(np.array(df), columns=self.columns)
            df.to_csv(fname, columns=self.columns, index=False)

    def dataload(self,pth, test_ratio=0.5, train=True):
        data = np.array(pd.read_csv(pth, header = None))[:-3]
        df1 = data[:-1, 0]
        df2 = data[1:, 0]
        diff = df2 - df1
        list_tmp = list(np.where(diff > 300000000 * 2)[0])
        list_tmp.insert(0, 0)
        #print(list_tmp)
        max_idx = np.argmax(np.diff(list_tmp))
        [idx_st, idx_ed] = list_tmp[max_idx ], list_tmp[max_idx + 1]
        #print(idx_st, idx_ed)
        self.saveObserved(pth, [idx_st, idx_ed])
        count = 1
        data_mbSize = []
        if train:
            for i in range(idx_st, idx_st + int((idx_ed - idx_st) * test_ratio)):
                
                if count % self.mb_size == 0:
                    yield np.array(data_mbSize).reshape(-1, 8, data.shape[1])
                    #print(len(data_mbSize))
                    data_mbSize = []
                data_mbSize.append(data[i:i + 8])
                count += 1 
                
            yield None
        else:
            data_mbSize = []
            for i in range(idx_st + int((idx_ed - idx_st) * test_ratio), idx_ed):
                #剩余的全送出去预测
                data_mbSize.append(data[i:i + 7])
            yield np.array(data_mbSize).reshape(-1, 7, data.shape[1])
        

    def trainday(self, is_training, anomaly, f):
        self.pointless = []
        pth = r'/home/wxh/lstmDnn/anomaly/pred_data/test/4665896876_28.csv'
        data = self.dataload(pth)
        
        raw_batch = next(data)
        cur_loss = sys.float_info.max
        endx = raw_batch.shape[1] - 1
        endt = raw_batch.shape[1]
        self.r2_s = []
        time_step_prd = 100
        data_dict = {
            'time':raw_batch[:, :, 0],
            'job':raw_batch[:, :, 2],
            'task':raw_batch[:, :, 3],
            'machine':raw_batch[:, :, 4],
            'x': raw_batch[:, :endx, 5:-3],
            't': raw_batch[:, 1:endt, 5:-3]
        }
        last_seq = 0
        #取一些test数据做验证看效果
        data_test = self.dataload(pth, train=False)
        next_data = next(data_test)
        data_dict['x'] = next_data[:-1, :, 5:-3]
        data_dict['t'] = next_data[1:, :, 5:-3]
        
        
        if self.train_test:
            
            for i in range(self.test_train_epoch):
                loss_mean = []
                data = self.dataload(pth)
                raw_batch = next(data)
                feature_size = raw_batch.shape[-1] - 8
                
                while raw_batch is not None:
                    
                    data_dict_test = {
                    'time':raw_batch[:, :, 0],
                    'job':raw_batch[:, :, 2],
                    'task':raw_batch[:, :, 3],
                    'machine':raw_batch[:, :, 4],
                    'x': raw_batch[:, :endx, 5:-3].reshape(-1, 7, feature_size),
                    't': raw_batch[:, 1:endt, 5:-3].reshape(-1, 7, feature_size)
                    }
                    if raw_batch is not None:
                        last_seq = data_dict_test['t'][-1].reshape(-1, 7, feature_size)
                        
                    _, cur_loss, pointloss, pred_y = self.model.train_step(data_dict_test, 
                    self.eval_tensors, update=True)
                    raw_batch = next(data)
                    
                    #loss_mean.append(pointloss)
                
                print('train loss ',cur_loss)

                #########################
                

                _, cur_loss, pointloss, pred_y = self.model.train_step(data_dict, 
                    self.eval_tensors, update=False)
                if self.earlyStop(cur_loss):
                    print('early stop is meet!')
                    break
                print('valid loss ', np.mean(pointloss,axis=0)[0])
                #print(np.mean(pointloss, axis = 0))
        #取最后一个batch的最后一组序列
        #data_dict['x'] = last_seq#这里搞得不对
        data_test = self.dataload(pth, train=False)
        next_data = next(data_test)
        data_dict['x'] = next_data[:-1, :, 5:-3]
        data_dict['t'] = next_data[1:, :, 5:-3]#预测时这个无用，随便给
        
        grount_truth = np.squeeze(data_dict['t'][:, -1, :])
        #revised 134-142 remove for i in time-step
        _, cur_loss, pointloss, pred_y = self.model.train_step(data_dict, self.eval_tensors, update=is_training)
        pred_y = pred_y[:, -1, :] # 1x7x7-> 1x1x7
        print('test loss ',cur_loss)
        pred_y = np.squeeze(pred_y)

        
        
        ground_truth_df = pd.DataFrame(grount_truth.squeeze(), columns=self.columns[5:])
        pred_df = pd.DataFrame(pred_y.squeeze(), columns=self.columns[5:])
        pred_df.to_csv(r'/home/wxh/lstmDnn/anomaly/pred_data/pred.csv', index = False)
        ground_truth_df.to_csv(r'/home/wxh/lstmDnn/anomaly/pred_data/ground_truth.csv', index = False)
        

    def earlyStop(self, loss):
        
        if loss >= self.loss:
            self.earlystop_count -= 1
        elif loss <= self.loss:
            self.loss = loss
            self.earlystop_count = 50
        if self.earlystop_count == 0:
            return True
        else:
            return False

    def load(self, jsonfile, outfile, logfile):
        tf.set_random_seed(408)
        np.random.seed(408)


        layer_list = [10]

        lr = 0.2e-1

        self.mb_size = 64
        self.maxbadcount = 100

        dataspecs = json.load(open(jsonfile, 'r'))
        
        sentence_length = dataspecs['sentence_length'] - 1
        self.sentence = sentence_length + 1
        token_set_size = dataspecs['token_set_size']
        embed_size = len(dataspecs['feature']) - 6
        self.datafeature = dataspecs['feature']
        self.resource = {'cpu': 0.9}
        self.columns=["ST", "ET", "jobID", "taskID", "machineID", "mean cpu usage"," canonical memory usage", 
                "disk io", "disk space", "CPI", "MAI"]
        self.bi = 0
        
        if self.bi==1:
            sentence_length += 1
            x = tf.placeholder(tf.float32, [None, sentence_length, embed_size]) # 8
            t = tf.placeholder(tf.float32, [None, sentence_length-2, embed_size]) # 6
        else:
            x = tf.placeholder(tf.float32, [None, sentence_length, embed_size]) # 7
            t = tf.placeholder(tf.float32, [None, sentence_length, embed_size]) # 7
        
        
        ph_dict = {'x': x, 't': t}
        token_embed = tf.Variable(tf.truncated_normal([token_set_size, embed_size]))

        token_losses, h_states, final_h, pred_y = cutils.bidir_lm_rnn_predict_pred(x, t, token_embed, layer_list)
        line_losses = tf.reduce_mean(token_losses, axis=1)
        avg_loss = tf.reduce_mean(line_losses)
        self.model = ModelRunner(avg_loss, ph_dict, learnrate=lr, decay_steps=1000)
        self.eval_tensors = [avg_loss, line_losses, pred_y]

        files = dataspecs['test_files']
        idx = 0
        self.saver = tf.train.Saver()
        self.test_train_epoch = 5000
        self.idx = idx
        self.train_test = 1
        self.trainday(False, False, files)




if __name__ == '__main__':
    jsonfile = r'/home/wxh/lstmDnn/anomaly/google_trace_config.json'
    outfile = r'/home/wxh/lstmDnn/anomaly/results_v2.csv'
    logfile = r'/home/wxh/lstmDnn/anomaly/loss_v2.log'
    m = model()
    m.load(jsonfile, outfile, logfile)