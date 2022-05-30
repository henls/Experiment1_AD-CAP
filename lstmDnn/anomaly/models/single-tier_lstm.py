#import tensorflow as tf
from sklearn.metrics import r2_score
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import json
import sys
import os
import datetime
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
        with open(r'/home/wxh/lstmDnn/anomaly/pkl/save.log', 'r') as f:
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
        
        raw_batch = data.next_batch()
        cur_loss = sys.float_info.max
        check_error = EarlyStop(self.maxbadcount)
        
        endx = raw_batch.shape[1] - 1
        
        
        endt = raw_batch.shape[1]
        training = check_error(raw_batch, cur_loss)
        self.r2_s = []
        while training:
            
            data_dict = {
                'time':raw_batch[:, :, 0],
                'job':raw_batch[:, :, 2],
                'task':raw_batch[:, :, 3],
                'machine':raw_batch[:, :, 4],
                'x': raw_batch[:, :endx, 5:-3],
                't': raw_batch[:, 1:endt, 5:-3]
            }
            
                

            _, cur_loss, pointloss = self.model.train_step(data_dict, self.eval_tensors, update=is_training)
            self.pointless.append(pointloss)
            #if not is_training:
            #self.write_results(('fixed', 'update')[is_training], data_dict, cur_loss, self.outfile, batch_num)
            batch_num += 1
            los.append(cur_loss)
            #if data_dict['time'][0] % 10 == 0:
            '''if batch_num % 5 == 0:
                print('%s %s %s %s %s %s' % (raw_batch.shape[0], data_dict['time'][0][0],
                                                ('fixed', 'update')[is_training],
                                                f, data.index, cur_loss))'''
            
            
            raw_batch = data.next_batch()
            
            training = check_error(raw_batch, cur_loss)
            #if raw_batch is not None:
            #    self.r2_score = 1 - (pointloss / np.var(raw_batch[:, 1:endt, 5:-3],axis=1))
            #print('r2score: ', np.max(self.r2_score, axis=0))
            #    self.r2_s.append(self.r2_score)
            if training < 0:
                return cur_loss
                exit(0)
        if np.mean(los) < self.loss:
            self.saver.save(self.model.sess, "/home/wxh/lstmDnn/anomaly/pkl/single-tier")
            self.loss = np.mean(los)
            print('model saved loss is ' + str(self.loss))
            with open(r'/home/wxh/lstmDnn/anomaly/pkl/save.log', 'a') as t:
                t.write(str(datetime.date.today()) + ' model saved loss is ' + str(self.loss) + '\n')
            try:
                self.r2_score = 1 - (pointloss / np.var(raw_batch[:, 1:endt, 5:-3],axis=1))
            except Exception as e:
                pass
        return los
            

    def write_results(self, is_training, data_dict, loss, outfile, batch):
            outfile.write('%s,%s,%s,%s \n' % (is_training, batch, data_dict['time'][0][0], loss))

    def load(self, jsonfile, outfile, logfile):
        tf.set_random_seed(408)
        #tf.random.set_seed(408)
        np.random.seed(408)

        #layer_list = [10]
        layer_list = [10]
        #lr = 1e-3
        #lr = 5e-3 04101614 cpu
        lr = 1e-3
        #embed_size = 20
        #embed_size = 1
        #self.mb_size = 64
        self.mb_size = 64
        self.maxbadcount = 10

        dataspecs = json.load(open(jsonfile, 'r'))
        #dataspecs = json.load(open('../safekit/features/specs/lm/test_config.json', 'r'))
        sentence_length = dataspecs['sentence_length'] - 1
        self.sentence = sentence_length + 1
        token_set_size = dataspecs['token_set_size']
        embed_size = len(dataspecs['feature']) - 8
        self.datafeature = dataspecs['feature']
        self.resource = {'cpu': 0.9}
        #x = tf.placeholder(tf.int32, [None, sentence_length])
        #t = tf.placeholder(tf.int32, [None, sentence_length])
        x = tf.placeholder(tf.float32, [None, sentence_length, embed_size])
        t = tf.placeholder(tf.float32, [None, sentence_length, embed_size])
        ph_dict = {'x': x, 't': t}
        token_embed = tf.Variable(tf.truncated_normal([token_set_size, embed_size]))

        token_losses, h_states, final_h = cutils.lm_rnn(x, t, token_embed, layer_list)
        
        line_losses = tf.reduce_mean(token_losses, axis=1)
        avg_loss = tf.reduce_mean(line_losses)
        
        #self.outfile = open(outfile, 'w')
        #self.outfile.write("status,batch,time,loss \n")

        
        self.model = ModelRunner(avg_loss, ph_dict, learnrate=lr)
        self.eval_tensors = [avg_loss, line_losses]

        files = dataspecs['test_files']
        
        idx = 0
        epoch = 10000
        #f = open(logfile, 'a')
        self.saver = tf.train.Saver()
        #for idx, f in enumerate(files[:-1]):
        while idx < epoch:
        
            loss = self.trainday(True, False, files)
            test_loss = self.trainday(False, False, files)
            #f.write('epoch {} loss {} min {} max {} tst_loss {} min{} max{} \n'.format(
            #    idx, np.mean(loss), np.min(loss), np.max(loss), 
            #    np.mean(test_loss), np.min(test_loss), np.max(test_loss)))
            print('epoch {} loss {} max {} tst_loss {} max {}'.format(
                idx, np.mean(loss), np.max(loss), 
                np.mean(test_loss), np.max(test_loss)))
            self.pointless = np.array(self.pointless)
            print('pointloss: ',np.max(self.pointless.reshape(-1,
                    self.pointless.shape[1], self.pointless.shape[2]), axis = (0, 1)))
            np.savetxt('/home/wxh/lstmDnn/anomaly/results/pointloss.csv',self.pointless.reshape(-1,
                    self.pointless.shape[2]) ,fmt='%f',delimiter=',')
            '''try:
                print('r2score: ', np.max(self.r2_score, axis=0))
                np.savetxt('/home/wxh/lstmDnn/anomaly/results/r2score.csv',np.array(self.r2_s).reshape(-1, sentence_length),fmt='%f',delimiter=',') 
            except Exception as e:
                pass'''
            idx += 1
        #self.outfile.close()




if __name__ == '__main__':
    jsonfile = r'/home/wxh/lstmDnn/anomaly/google_trace_config.json'
    outfile = r'/home/wxh/lstmDnn/anomaly/results_v2.csv'
    logfile = r'/home/wxh/lstmDnn/anomaly/loss_v2.log'
    m = model()
    m.load(jsonfile, outfile, logfile)