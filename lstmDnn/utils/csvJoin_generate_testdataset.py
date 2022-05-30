# -*- coding:utf-8 -*-
import dask.dataframe as dd
import glob
import os
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import sys
sys.path.append(r'/home/wxh/google_trace/codes')
from tools import *
from mpi4py import MPI
import re
'''
合并task_trace中数据:并连接成一个完整的时序数据:最后三列增加了对资源的需求:cpu\ram\disk
'''

class join():

    def __init__(self, key, value, path, eventPth, draw):
        self.key = key
        self.value = value
        self.path = path
        self.eventPth = eventPth
        self.draw = draw

    def sort_custom(self, l):
        int_list = re.findall('\d+', os.path.basename(l))
        int_list = [str(int(i)) for i in int_list]
        return int(''.join(int_list))
    def start(self):
        savepth = self.path + '_unbroken/' + str(self.key)
        fname = '/'.join([savepth, str(self.value) + '.csv'])#训练集的位置
        savepth = self.path + '_unbroken_testDateset/' + str(self.key)
        newfname = '/'.join([savepth, str(self.value) + '.csv'])
        if os.path.exists(fname) == 0:#排除训练集中有的数据
            #print(self.path + '/' + str(self.key) + '/' + str(self.key) + '_' + str(self.value) + '-*')
            cfiles = sorted(glob.glob(self.path + '/' + str(self.key) + '/' + str(self.key) + '_' + str(self.value) + '-*'))
            cfiles.sort(key = self.sort_custom)
            first = dd.read_csv(cfiles[0], header = None)
            for i in cfiles[1:]:
                try:
                    nxt = dd.read_csv(i, header = None)
                    first = dd.concat([first, nxt], axis = 0)
                except Exception as e:
                    pass
            first = first.compute()
            eventFiles = sorted(glob.glob(self.eventPth + '/*'))
            for i in eventFiles[::-1]:
                pr = pd.read_csv(i, header=None)
                matched = pr[pr.iloc[:, 2] == self.key]
                if len(matched) != 0:
                    requestCpu = matched.iloc[len(matched)-1, 9]
                    requestRam = matched.iloc[len(matched)-1, 10]
                    requestDisk = matched.iloc[len(matched)-1, 11]
                    break
            cpuRelated = [5, 13]
            ramRelated = [6, 7, 8, 9, 10]
            diskRelated = [12]
            first.iloc[:, 15] = 1 / first.iloc[:, 15]
            '''for i in cpuRelated:
                first.iloc[:, i] /= requestCpu'''
            first.insert(first.shape[1], first.shape[1], requestCpu)
            '''for i in ramRelated:
                first.iloc[:, i] /= requestRam'''
            first.insert(first.shape[1], first.shape[1], requestRam)
            '''for i in diskRelated:
                first.iloc[:, i] /= requestDisk'''
            first.insert(first.shape[1], first.shape[1], requestDisk)
            try:
                os.makedirs(savepth)
            except Exception as e:
                pass
            
            first.to_csv(newfname, index = 0, header = 0)
            y_label = [r'cpu usage %', r'memory usage %', r'memory usage %', r'memory usage %',
                        r'disk io time', r'disk space usage %', r'cpu usage %', r'CPI', r'MAI']
            title = [r'mean cpu usage', r'canonical memory usage', r'unmapped page cache memory usage',
                        r'total page cache memory usage', r'mean disk io time', r'mean local disk space used', r'max cpu usage',
                        r'CPI', r'MAI']
            data_range = [5, 6, 8, 9, 11, 12, 13, 15, 16]
            if self.draw:
                plt.subplots(figsize=(18, 10))
                for i in range(1,10):
                    plt.subplot(3,3,i)
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)
                    plt.xlabel('time')
                    plt.ylabel(y_label[i - 1])
                    plt.title(title[i - 1])
                    plt.plot(range(len(first)), first.iloc[:, data_range[i - 1]])
                f = plt.gcf()
                f.savefig(savepth + '/' + str(len(first)) + '-' + str(self.value) + '.png')
                f.clear()
            print(savepth + '   saved...')
        else:
            print('Exist skip....')


def joinMulti(dicts, key_amount, value_amount):
    path = '/home/wxh/google_trace/task_trace'
    eventPth = '/home/wxh/google_trace/task_events'
    slice_ = key_amount // size
    choose_key = random.sample(dicts.keys(), key_amount)
    if rank != size - 1:
        choose_key = choose_key[slice_*rank:slice_*(rank+1)]
    else:
        choose_key = choose_key[slice_*rank:]
    print('choose_keys: ', choose_key)
    for key in choose_key:
        if value_amount <= len(dicts[key]):
            choose_value = random.sample(dicts[key], value_amount)
        else:
            choose_value = dicts[key]
        print('choose_values: ', choose_value)
        for v in choose_value:
            test = join(key, v, path, eventPth, draw=False)
            try:
                test.start()
            except Exception as e:
                pass
    print('process id: {} over....'.format(rank))
            

def get_dicts_of_trace(trace_pth):
    files = glob.glob(trace_pth + '/*/*.csv')
    dict_new = {}
    for i in files:
        k = int(os.path.basename(i).split('_')[0])
        v = int(os.path.basename(i).split('_')[1].split('-')[0])
        try:
            dict_new[k].append(v)
        except Exception as e:
            dict_new[k] = [v]
    return dict_new
                

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #dicts = loadDict('/home/wxh/google_trace/task_dict/totalDict')
    dicts = get_dicts_of_trace(r'/home/wxh/google_trace/task_trace')
    key_amount = 500
    value_amount = 20
    random.seed(2)
    joinMulti(dicts, key_amount, value_amount)
