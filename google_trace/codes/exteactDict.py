#合并全部字典，并用mpi匹配全时段数据
from ast import Is
from json import load
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import glob as glob
import os
import time
from tools import *
from mpi4py import MPI


class unbrokenDict():

    def __init__(self, dictPth, usagePth, savePth):
        self.dictPth = dictPth
        self.usagePth = usagePth
        self.savePth = savePth

    def loadAllDict(self):
        dicts = sorted(glob.glob(self.dictPth))
        self.dict = dicts


    def joinDict(self):
        if os.path.exists('/'.join(dictPth.split('/')[:-1]) + '/totalDict') == 0:
            self.totalDict = loadDict(self.dict[0])
            print('Join dict...')
            for i in self.dict[1:]:
                newDict = loadDict(i)
                sameKeys = self.totalDict.keys() & newDict.keys()
                for j in sameKeys:
                    self.totalDict[j] = self.totalDict[j].union(newDict[j])
                    newDict.pop(j, None)
                self.totalDict.update(newDict)
            print('Join dict complete...')
            saveDict(self.totalDict, '/'.join(dictPth.split('/')[:-1]) + '/totalDict')
            print('Saved totalDict...')
        else:
            self.totalDict = loadDict('/'.join(dictPth.split('/')[:-1]) + '/totalDict')
            print('load totalDict...')

    def loadUsageFile(self):
        self.usageFile = sorted(glob.glob(self.usagePth))
        
    def joinCSV(self, spans):
        start = time.time()
        finStep = 1
        for i in spans:
            s = time.time()
            f = self.usageFile[i]
            openCSV= dd.read_csv(f, header=None, dtype={19: 'float64'}).iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19]]
            step = 0
            for k, v in self.totalDict.items():
                tt = openCSV[openCSV.iloc[:, 2] == k]
                s2 = time.time()
                step3 = 0
                for task in v:
                    s3 = time.time()
                    try:
                        os.mkdir(r'/home/wxh/google_trace/task_trace/' + str(k))
                    except Exception as e:
                        pass
                    partitions = tt[tt.iloc[:, 3] == task].npartitions
                    Isexist = []
                    for xx in range(partitions):
                        Isexist.append(os.path.exists(r'/home/wxh/google_trace/task_trace/' + str(k) + '/' +str(k) 
                        + '_' + str(task) + '-' + str(i) + '-' + str(xx) + '.csv'))
                    if sum(Isexist) != partitions:
                        tt[tt.iloc[:, 3] == task].to_csv(r'/home/wxh/google_trace/task_trace/' + str(k) + '/' +str(k) 
                        + '_' + str(task) + '-' + str(i) + '-*.csv', index=0, header=0)
                    else:
                        print('partitions is {} Files exist pass...'.format(partitions))
                    end3 = time.time()
                    step3 += 1
                    print('task ' + str(task) + ' ' + str(round(step3/len(v)*100, 2)) + 
                    '%' + ' Expect time ' + transform((end3 - s3) / (1/len(v)) - end3 + s2))
                end2 = time.time()
                step += 1
                print('keys ' + str(k) + ' ' + str(round(step/len(self.totalDict.keys())*100, 2)) + '%' + ' Expect time ' + transform((end2 - s2) / (1/len(self.totalDict.keys())) - end2 + s))
            end = time.time()
            print('files ' + str(i) + ' ' + str(round(finStep/len(spans)*100, 2)) + '%' + ' Expect time ' + transform((end - s) / (1/len(spans)) - end + start))
            finStep += 1

    def start(self, spans):
        self.loadAllDict()
        self.joinDict()
        self.loadUsageFile()
        self.joinCSV(spans)
dictPth = r'/home/wxh/google_trace/task_dict/jobTaskDict_*'
usagePth = r'/home/wxh/google_trace/task_usage/*.csv'
savePth = r'/home/wxh/google_trace/task_trace'
t = unbrokenDict(dictPth, usagePth, savePth)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    task_file = list(range(23))
    task_size = len(task_file)
    block = task_size // size
    task_list = []
    for i in range(size-1):
        task_list.append(task_file[i * block : (i + 1) * block])
    task_list.append(task_file[(size-1)*block:])
else:
    task_list = None

spans = comm.scatter(task_list, root = 0)
t.start(spans)