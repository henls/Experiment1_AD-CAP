# -*- coding:utf-8 -*-
import glob
import pandas as pd
import numpy as np
import os

def pickItemsFromCSV(length, savedir, DataPth):
    '''
    从csv中挑选时序序列长的数据并移动到savedir
        param:
            length:要求的序列最短长度
            savedir:保存的文件夹
            DataPth:源文件的位置
        return->none
    '''
    dirlist = glob.glob(DataPth+'/*/*')
    dicts = {}
    max_length = 0
    max_fname = ''
    for i in dirlist:
        data = pd.read_csv(i, header=None)
        df1 = data.iloc[:-1,0]
        df2 = data.iloc[1:,0]
        df_diff = np.array(df2) - np.array(df1)
        idx = (df_diff>300000000)
        idx_break = np.where(idx==True)[0]
        idx_break = list(idx_break)
        idx_break.insert(0,0)
        
        try:
            max_seq = np.max(np.diff(idx_break))
        except Exception as e:
            continue
        max_idx_seq = np.argmax(np.diff(idx_break)) + 1
        dicts[i] = [max_seq, idx_break[max_idx_seq]]
        if dicts[i][0] > max_length:
            max_length = dicts[i][0]
            max_fname = i
    print('最大序列长度是{}, 对应 {} 文件'.format(max_length, max_fname))
    savename = savedir + '/' + '/'.join(max_fname.split('/')[-2:])
    try:
        os.makedirs(os.path.dirname(savename))
    except Exception as e:
        pass
    os.system('mv ' + max_fname + ' ' + savename)
    print('mv ' + max_fname + ' ' + savename)

if __name__ == '__main__':
    length = 100
    savedir = r'/home/wxh/google_trace/cache/testData'
    DataPth = r'/home/wxh/google_trace/task_trace_unbroken_testDateset'
    pickItemsFromCSV(length, savedir, DataPth)

