import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import glob as glob
import os
import time
from tools import *
from mpi4py import MPI

totalDict = loadDict(r'/home/wxh/google_trace/task_dict/totalDict')
first = dd.read_csv(r'/home/wxh/google_trace/task_usage/part-00000-of-00500.csv', header=None, dtype={19: 'float64'})
'''
for k, v in totalDict.items():
    if len(v)>300:
        k_csv = first[first.iloc[:, 2] == k]
        start = time.time()
        for v_task in v:
            v_csv = k_csv[k_csv.iloc[:, 3] == v_task]
            v_csv.compute()
            print('single find: ' + str(time.time() - start)  + 'length: ' + str(len(v)))
            break
        a = (time.time()-start)*len(v)
        print(transform(a*len(totalDict) - time.time() + start))
        break

for k, v in totalDict.items():
    if len(v) > 300:
        k_csv = first[first.iloc[:, 2] == k]
        start = time.time()
        v_csv = k_csv[k_csv.iloc[:, 3].isin(v)]
        v_csv.compute()
        print('multi find: ' + str(time.time() - start) + 'length: ' + str(len(v)))
        print(transform((time.time() - start)*len(totalDict) - time.time() + start))
        break'''

start = time.time()
k = totalDict.keys()
max_len = 0
for i in k:
    if len(totalDict[i]) > max_len:
        max_len = len(totalDict[i])
        v = totalDict[i]
first = first.compute().iloc[:1000]
y = first[first.iloc[:, 2].isin(k)]
x_2 = []
for i in v:
    #x_2.append(y.iloc[:, 3].isin([i]))
    x_2.append(np.diag(y.iloc[:, 3].isin([i])))
x_2 = np.array(x_2).reshape(len(v), len(y), len(y))
choose_matrix = x_2
k_csv = np.matmul(x_2, np.array(y))

print(k_csv[0])
np.savetxt('/home/wxh/google_trace/matmul.csv',k_csv[0],fmt='%f',delimiter=',')
a = k_csv.groupby([3])
#v_csv.to_csv(r'/home/wxh/google_trace/maxlength/*.csv', index=0, header=0)
for i in range(1608):
    c = a.get_group(i)
    print(c.compute())

print(time.time() - start)
print(max_len)