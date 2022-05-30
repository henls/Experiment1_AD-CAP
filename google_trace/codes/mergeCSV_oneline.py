#coding=utf-8
from tools import *
import pandas as pd
import glob
import os
import numpy as np
import random
import re
from multiprocessing import Process
import time


def main(totalDict, usagePth, pickLength, savePth):
    dicts = loadDict(totalDict)
    [keys, values] = pickKeysAndValues(dicts, pickLength)
    mergeUsageTrace(keys, values, savePth, usagePth)

def pickKeysAndValues(dicts, pickLength):
    '''keys = random.sample(dicts.keys(), pickLength)
    values = []
    for i in keys:
        values += random.sample(dicts[i], 1)'''
    #keys = [1863690462] + [4665896876]*8 + [6162908962] #旧的trace
    #values = [1, 28, 43, 95, 107, 173, 183, 194, 227, 0] #旧的trace
    keys = [3727144341]*2 + [3769734721]*2
    values = [2, 8, 0, 2]

    return keys, values

def mergeUsageTrace(keys, values, savePth, usagePth):
    usageList = loadUsageTrace(usagePth)
    for k, v in zip(keys, values):
        P = Process(target=matchKeysAndValues, args=(usageList, k, v, savePth,))
        #matchKeysAndValues(usageList, k, v, savePth)
        #time.sleep(1)
        P.start()

def matchKeysAndValues(usageList, k, v, savePth):
    fragment = []
    start = time.time()
    expectTime = '**'
    for csvPth in usageList:
        nowtime = time.time()
        print('INFO: Processing key#{} value#{} files# {} remainingTime {}'.format(k, v, csvPth, expectTime))
        df = pd.read_csv(csvPth, header=None)
        slice_k = df[df.iloc[:, 2] == k]
        slice_v = slice_k[slice_k.iloc[:, 3] == v]
        if len(slice_v) != 0:
            fragment.append(slice_v)
        elif len(fragment) != 0 and len(slice_v) == 0:
            #表示从前往后找已经找到全部的记录，后面不需要找了
            break
        end = time.time()
        expectTime = transform((end - nowtime) * len(usageList) - (end - start))

    first = fragment[0]
    print('INFO: concating key#{} value#{}'.format(k, v))
    for i in fragment[1:]:
        first = pd.concat([first, i])
    try:
        os.makedirs(savePth)
    except Exception as e:
        pass
    savePth = savePth + '/' + str(k) + '_' + str(v) + '_' + str(len(first)) + '_.csv'
    columns=["ST", "ET", 
             "jobID", "taskID",
             "mean cpu usage", "canonical memory usage",
             "disk io", "disk space"]
    pickColumns = first.iloc[:, [0, 1, 2, 3, 5, 6, 11, 12]]
    pickColumns.columns = columns
    pickColumns.to_csv(savePth, index=False)

def loadUsageTrace(pth):
    result = sorted(glob.glob(pth + '/*.csv'))
    result.sort(key = sort_custom)
    return result

def sort_custom(l):
    int_list = re.findall('\d+', os.path.basename(l))
    int_list = [str(int(i)) for i in int_list]
    return int(''.join(int_list))

if __name__ == '__main__':
    random.seed(400)
    totalDict = r'/home/wxh/google_trace/task_dict/totalDict'
    usagePth = r'/home/wxh/google_trace/task_usage'
    pickLength = 10
    savePth = r'/home/wxh/transformer/data'
    main(totalDict, usagePth, pickLength, savePth)
    exit(0)
    k = 4665896876
    v = 28
    pth = r'/home/wxh/google_trace/task_usage'
    usageList = loadUsageTrace(pth)[:2]
    savePth = r'/home/wxh/transformer/data'
    matchKeysAndValues(usageList, k, v, savePth)