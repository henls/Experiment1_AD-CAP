#1.一次处理全部时间长度的单条数据，让数据跑出一条就能用一条
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import glob as glob
import os
import time
from tools import *
from mpi4py import MPI


class trackGoogle():

    def __init__(self, workDir, usageFileP, typeFileP):
        self.usageFileP = usageFileP
        self.typeFileP = typeFileP
        self.workDir = workDir
    
    def loadUsageFile(self):
        self.usageFiles = sorted(glob.glob(self.usageFileP))

    def initTypeFiles(self):
        self.typeFiles = sorted(glob.glob(self.typeFileP))
        print('Initing...')

    def loadEventFile(self, idx):
        self.typeFiles = sorted(glob.glob(self.typeFileP))
        print('reading typeFilees...')
        self.typeFileDataframe = dd.read_csv(self.typeFiles[idx], header = None, dtype={0:'float64', 4: 'float64'})
        print('typeFiles read finished...')

    def findTopkTrace(self, spans):
        for idx in spans:
            jobTaskDict = {}
            self.loadEventFile(idx)
            TaskDictpath = self.workDir + 'jobTaskDict_' + str(idx)
            if os.path.exists(TaskDictpath) == 0:
                tmp = self.typeFileDataframe[self.typeFileDataframe.iloc[:,7] == 3]
                jobKeys = set(tmp.iloc[:, 2].compute())
                start  = time.time()
                finStep = 1
                for i in jobKeys:
                    s = time.time()
                    tt = tmp[tmp.iloc[:, 2] == i]
                    tt = tt.iloc[:, 3]
                    jobTaskDict[i] = set(tt.compute())
                    end = time.time()
                    print('jobKey ' + str(i) + ' ' + str(round(finStep/len(jobKeys) * 100, 2)) + '%' + ' Expect time ' + transform((end - s) / (1/len(jobKeys)) - end + start))
                    finStep += 1
                saveDict(jobTaskDict, TaskDictpath)
                print('Dict saved...')
            else:
                print('Find Dict reloading...')
                jobTaskDict = loadDict(TaskDictpath)
                print('Reload successfully...')
            print('file ' + self.typeFiles[idx] + ' dict built...')
    
    def ToUnbrokenTrace(self, csvFile):
        pass


    def start(self, spans):
        print('rank ' + str(rank) + ': range ' + str(spans[0]) + ' to ' + str(spans[-1]) + ' size: ' + str(len(spans)))
        self.initTypeFiles()
        self.findTopkTrace(spans)

workDir = r'/home/wxh/google_trace/task_dict/'
usageFiles = r'/home/wxh/google_trace/task_usage/*.csv'
typeFiles = r'/home/wxh/google_trace/task_events/*.csv'
#typeFiles = r'/home/wxh/google_trace/task_events/part-0000*-of-00500.csv'
#typeFiles = r'/home/wxh/google_trace/codes/part-0000*-of-00500.csv'
d1 = trackGoogle(workDir, usageFiles, typeFiles)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
if rank == 0:
    d1.initTypeFiles()
    task_file = list(range(140))
    task_size = len(task_file)
    block = task_size // size
    task_list = []
    for i in range(size-1):
        task_list.append(task_file[i * block : (i + 1) * block])
    task_list.append(task_file[(size-1)*block:])
else:
    task_list = None

spans = comm.scatter(task_list, root = 0)
d1.start(spans)