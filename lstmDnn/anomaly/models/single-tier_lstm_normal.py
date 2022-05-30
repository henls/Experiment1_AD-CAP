#import tensorflow as tf
from sklearn.metrics import r2_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import json
import sys
import os
import datetime
import time
sys.path.append(r'/home/wxh/lstmDnn/anomaly')
sys.path.append(r'/home/wxh/lstmDnn/anomaly/safekit')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
    def trainday(self, is_training, anomaly, f):
        los = []
        self.pointless = []
        batch_num = 0
        data = cutils.OnlineBatcher_custom_mem(f, self.mb_size, self.sentence, 
                                            self.datafeature, is_training, 
                                            self.resource, anomaly, delimiter=',', 
                                            skipheader=False, norm=False)
        if data.saved == 0:
            raw_batch = data.next_batch()
            cur_loss = sys.float_info.max
            
            
            endx = raw_batch.shape[1] - 1
            endt = raw_batch.shape[1]
            training = 1
            self.r2_s = []
            while training:
                
                data_dict = {
                    'time':raw_batch[:, :, 0],
                    'job':raw_batch[:, :, 2],
                    'task':raw_batch[:, :, 3],
                    'machine':raw_batch[:, :, 4],
                    'x': raw_batch[:, :endx, 5:-5],
                    't': raw_batch[:, 1:endt, 5:-5]
                }
                if self.bi == 1:
                    data_dict['x'] = raw_batch[:, :endx + 1, 5:-5]
                    data_dict['t'] = raw_batch[:, 1:endt-1, 5:-5]
                
                    

                _, cur_loss, pointloss = self.model.train_step(data_dict, self.eval_tensors, update=is_training)
                self.pointless.append(pointloss)

                batch_num += 1
                los.append(cur_loss)
                
                raw_batch = data.next_batch()
                #training = check_error(raw_batch, cur_loss)
                if raw_batch is None:
                    training = 0
                #if raw_batch is not None:
                #    self.r2_score = 1 - (pointloss / np.var(raw_batch[:, 1:endt, 5:-3],axis=1))
                #print('r2score: ', np.max(self.r2_score, axis=0))
                #    self.r2_s.append(self.r2_score)
                if training < 0:
                    return cur_loss
                    exit(0)
            if np.mean(los) < self.loss:
                self.saver.save(self.model.sess, "/home/wxh/lstmDnn/anomaly/pkl/single-tier-normal")
                self.loss = np.mean(los)
                print('model saved loss is ' + str(self.loss))
                with open(r'/home/wxh/lstmDnn/anomaly/pkl/save_normal.log', 'a') as t:
                    t.write(str(datetime.date.today()) + ' model saved loss is ' + str(self.loss) + '\n')
            return los
        else:
            raw_batch = data.data[0]
            cur_loss = sys.float_info.max

            endx = raw_batch.shape[1] - 1
            endt = raw_batch.shape[1]

            dataset_len = len(data.data)
            self.r2_s = []
            x = data.data[:, :, :endx, 5:-5]
            t = data.data[:, :, 1:endt, 5:-5]
            #t = x
            counter = 0
            while counter < dataset_len:
                
                data_dict = {
                    'x': x[counter],
                    't': t[counter]
                }

                _, cur_loss, pointloss = self.model.train_step(data_dict, self.eval_tensors, update=is_training)
                if self.idx % self.freq_print == 0:
                    self.pointless.append(pointloss)
                    los.append(cur_loss)
                
                counter += 1
            if self.idx % self.freq_print == 0 and is_training == True:
                if np.mean(los) < self.loss:
                    self.saver.save(self.model.sess, "/home/wxh/lstmDnn/anomaly/pkl/single-tier-normal-feature-4")
                    self.loss = np.mean(los)
                    print('model saved loss is ' + str(self.loss))
                    with open(r'/home/wxh/lstmDnn/anomaly/pkl/save_normal.log', 'a') as t:
                        t.write(str(datetime.date.today()) + ' model saved loss is ' + str(self.loss) + '\n')
            return los

    def write_results(self, is_training, data_dict, loss, outfile, batch):
            outfile.write('%s,%s,%s,%s \n' % (is_training, batch, data_dict['time'][0][0], loss))


    def load(self, jsonfile, outfile, logfile):
        tf.set_random_seed(408)
        #tf.random.set_seed(408)
        np.random.seed(408)

        #layer_list = [10]
        layer_list = [15]
        self.layers_num = layer_list[0]
        lr = 1e-3
        #lr = 5e-3 04101614 cpu
        #lr=1e-2 4-15
        #lr = 4e-4 4-17
        #lr = 0.7e-7 #saved with 6 feature
        #embed_size = 20
        #embed_size = 1
        #self.mb_size = 64
        self.mb_size = 64
        self.maxbadcount = 100

        dataspecs = json.load(open(jsonfile, 'r'))
        
        sentence_length = dataspecs['sentence_length'] - 1
        self.sentence = sentence_length + 1
        token_set_size = dataspecs['token_set_size']
        embed_size = len(dataspecs['feature']) - 8
        self.datafeature = dataspecs['feature']
        self.resource = {'cpu': 0.9}
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
        if self.bi == 1:
            token_losses, h_states, final_h = cutils.bidir_lm_rnn(x, t, token_embed, layer_list)
        else:
            #token_losses, h_states, final_h = cutils.lm_rnn(x, t, token_embed, layer_list)
            token_losses, h_states, final_h = cutils.bidir_lm_rnn_predict(x, t, token_embed, layer_list)
        line_losses = tf.reduce_mean(token_losses, axis=1)
        avg_loss = tf.reduce_mean(line_losses)
        


        #5954
        #
        self.model = ModelRunner(avg_loss, ph_dict, learnrate=lr, decay_steps=7000)
        self.eval_tensors = [avg_loss, line_losses]

        files = dataspecs['test_files']
        
        idx = 0
        epoch = 10000
        
        self.saver = tf.train.Saver()
        self.freq_print = 1
        while idx < epoch:
            self.idx = idx
            loss = self.trainday(True, False, files)
            test_loss = self.trainday(False, False, files)
            
            if idx % self.freq_print == 0:
                print('epoch {} loss {} max {} tst_loss {} max {}'.format(
                    idx, np.mean(loss), np.max(loss), 
                    np.mean(test_loss), np.max(test_loss)))
                self.pointless = np.array(self.pointless)
                print('pointloss max: ',np.max(self.pointless.reshape(-1,
                        self.pointless.shape[1], self.pointless.shape[2]), axis = (0, 1)))
                print('pointloss mean: ',np.mean(self.pointless.reshape(-1,
                        self.pointless.shape[1], self.pointless.shape[2]), axis = (0, 1)))
            
            
            idx += 1
        




if __name__ == '__main__':
    jsonfile = r'/home/wxh/lstmDnn/anomaly/google_trace_config.json'
    outfile = r'/home/wxh/lstmDnn/anomaly/results_v2.csv'
    logfile = r'/home/wxh/lstmDnn/anomaly/loss_v2.log'
    m = model()
    m.load(jsonfile, outfile, logfile)