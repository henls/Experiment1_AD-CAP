from email import header
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask
import glob as glob
import os
import time

def transform(num):
    h = num // 3600
    m = num % 3600 //60
    s = num % 3600 % 60
    if h == 0:
        h = ''
    else:
        h = str(int(h)) + 'h'
    if m == 0:
        m = ''
    else:
        m = str(int(m)) + 'm'
    if s == 0:
        s =''
    else:
        s = str(int(s)) + 's'
    return h+m+s
csvFIles = sorted(glob.glob(r'/home/wxh/google_trace/task_usage/*.csv'))
result = dd.read_csv(csvFIles[0], header=None, dtype={19: 'float64'}).iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19]]
pd_result = result.compute()
job_key = set(pd_result.iloc[:, 2])
key_dict = {}
for i in job_key:
    key_dict[i] = set(pd_result[pd_result.iloc[:, 2] == i].iloc[:,3])
save_step = 5
for i in range(1,len(csvFIles)):
#for i in range(1,2):
    next = dd.read_csv(csvFIles[i], header=None)
    result = dd.concat([result, next], axis = 0)
    pd_result = next.compute()
    job_key = set(pd_result.iloc[:, 2])
    for j in job_key:
        key_dict[j] = set(pd_result[pd_result.iloc[:, 2] == j].iloc[:,3])
    print(str(i) + '  ' + csvFIles[i].split('/')[-1] + '  concated...')
    if i % 5 == 0:
        counter = 0
        total_time = time.time()
        for k, v in key_dict.items():
            start = time.time()
            tmp = result[result.iloc[:, 2] == k]
            for idx, jj in enumerate(v):
                csv_concat = tmp[tmp.iloc[:, 3] == jj]
                to_csv = '/home/wxh/google_trace/task_trace/' + str(len(csv_concat)) + '_' + str(k) + '_' + str(jj) + '-' + str(i)
                try:
                    os.mkdir(to_csv)
                except Exception as e:
                    pass
                csv_concat.to_csv(to_csv + '/'  + '*.csv', index=0, header=0)
                end = time.time()
                print(to_csv.split('/')[-1] + '  write... ' + str(round(idx/len(v) * 100, 2)) + '%' + '  '
                 + str(round(counter/len(key_dict.keys())*100, 2)) + '%')
            print('   expected time: ' + str(transform((end-start) / (1/len(key_dict.keys())*100) - end + total_time)))
            counter += 1


#df = pd.read_csv(path, header=None)
#df.to_csv('/home/wxh/google_trace/task_usage/test.csv', header=0, index=0, na_rep='NULL')
