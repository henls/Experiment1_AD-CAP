# -*- coding: utf-8 -*-
from matplotlib.pyplot import axis
import pandas as pd
from yaml import load, Loader, dump
import datetime
import os
import sys
import glob
import numpy as np
import time


class data_regular(object):

    def __init__(self, config_yaml):
        with open(config_yaml, 'r') as f:
            stream = f.read()
        config = load(stream, Loader=Loader)
        self.raw_data_app = config['raw_data']['app']
        self.raw_data_resource = config['raw_data']['resource']
        tmp = self.raw_data_resource.split('/')[-2]
        self.savedir = config['proceed_data']['savedir'] + '/' + tmp
        self.interval = config['interval']
        self.average = config['average']
        self.fill_path = config['filled_data']['dir'] + '/' + tmp
        self.proceed_path = config['proceed_data']['savedir'] + '/' + tmp
        self.align = config['align_data']['dir'] + '/' + tmp
        self.createFolds()

    def txt2csv(self, config_yaml):
        dataList = glob.glob(self.raw_data_app + '/*')
        for i in dataList:
            self.fillData(i, self.fill_path)
        data_merge = self.session_merge(self.fill_path + r'/session-fill.txt', \
        self.interval)
        self.save_csv(data_merge)
        self.csv_normal()

    def csv_normal(self):
        csvframe = self.raw_df
        frame_np = np.array(csvframe.iloc[:, 1:])
        frame_norm = np.zeros(frame_np.shape)
        frame_norm[:, 0] = frame_np[:, 0] / 100
        frame_norm[:, 1:] = (frame_np[:, 1:] - np.min(frame_np[:, 1:], axis = 0)) / (
                            np.max(frame_np[: ,1:], axis = 0) - np.min(frame_np[: ,1:], 
                            axis = 0)
                        )
        frame_norm[:, 5] = frame_np[:, 5] / 100
        frame_norm = np.round(frame_norm, 4)
        df = pd.DataFrame(frame_norm, columns=['cpu_usage%', 'io_stps', 
            'io_wtps', 'io_bread/s', 'io_bwrtn/s',
            'mem_used%', 'mem_kbbuffers', 'mem_kbcached', 
            'session_num'])
        csvframe.loc[:, 'cpu_usage%':'session_num'] = frame_norm
        save_path = self.proceed_csv.split('.')[0] + '-normal.xlsx'
        save_path_csv = self.proceed_csv.split('.')[0] + '-normal.csv'
        #df.to_csv(save_path , sep = ",", index = False)
        df.to_excel(save_path , index = False)
        csvframe.to_csv(save_path_csv, sep = ",", index = False)


    def fillData(self, txt_file, fill_path):
        with open(txt_file, 'r') as f:
            sequences = f.readlines()
        with open(txt_file, 'r') as f:
            content = f.read()
        tmp_txt = []
        length_fill = 0
        last= sequences[-1].split(' ')[0]
        first= sequences[0].split(' ')[0]
        last = datetime.datetime.strptime('2022-1-1 ' + last, '%Y-%m-%d %H:%M:%S')
        first = datetime.datetime.strptime('2022-1-1 ' + first, '%Y-%m-%d %H:%M:%S')
        length_full = (last - first).seconds
        length_actual = len(sequences)
        record_time = 0
        fill_flag = 0
        pointer = 0
        while length_full != record_time:
            record_time += 1
            new_item = sequences[pointer]
            tmp_txt.append(new_item)
            if pointer < length_actual - 1:
                next_item = sequences[pointer + 1]
            else:
                break
            [hour, minu, sec] = new_item.split(' ')[0].split(':')
            rt_count = ' '.join(new_item.split(' ')[1:])
            next_stamp = self.next_time(hour, minu, sec)
            while next_item.split(' ')[0] != next_stamp:
                [hour, minu, sec] = new_item.split(' ')[0].split(':')
                rt_count = ' '.join(new_item.split(' ')[1:])
                next_stamp = self.next_time(hour, minu, sec)
                if next_stamp not in content:
                    length_fill += 1
                    tmp_txt.append(next_stamp + ' ' + rt_count)
                    fill_flag = 1
                new_item = next_stamp + ' ' + rt_count
            pointer += 1
        print('file: {}, total: {}, actual: {}, fill:{}'.format(os.path.basename(txt_file), length_full, length_actual, length_fill))
        try:
            os.mkdir(fill_path)
        except Exception as e:
            pass
        if fill_flag == 1:
            fill_file_name = fill_path + '/' + os.path.basename(txt_file).split('.')[0] + '-fill.txt'
            with open(fill_file_name, 'w') as f:
                f.writelines(tmp_txt)
        else:
            fill_file_name = fill_path + '/' + os.path.basename(txt_file).split('.')[0] + '-fill.txt'
            with open(fill_file_name, 'w') as f:
                f.writelines(sequences)
        

    def next_time(self, hour, minu, sec):
        hour, minu, sec  = int(hour), int(minu), int(sec)
        sec_next = sec + 1
        minu_next = minu
        hour_next = hour
        if sec_next >= 60:
            sec_next -= 60
            minu_next = minu + 1
        if minu_next >= 60:
            minu_next -= 60
            hour_next = hour + 1
        if hour_next >= 24:
            hour_next -= 24
        return str(hour_next).zfill(2) + ':' + str(minu_next).zfill(2) + ':' + str(sec_next).zfill(2)


    def session_merge(self, sess_file, interval):
        with open(sess_file, 'r') as f:
            sess_cont = f.readlines()
        startPoint = r'02:50:03'
        with open(self.raw_data_resource + r'/client0_cpu', 'r') as f:
            cpu = f.readlines()
        cpu_dict = {}
        self.IsAM = cpu[0].split()[1] == 'AM'
        align_path = self.timeAlign(sess_file)
        with open(align_path, 'r') as f:
            session = f.readlines()
        sess_dict = {}
        for i in session:
            [k, v] = i.split(' ')
            v = v.strip()
            sess_dict[k] = v
        #sess_dict = self.merge(sess_dict, startPoint, interval)
        #convert to dict
        
        for i in cpu:
            [k, v] = i.split()[0], str(100 - float(i.split()[-1]))
            cpu_dict[k] = v
        with open(self.raw_data_resource + r'/client0_io', 'r') as f:
            io = f.readlines()
        io_dict = {}
        for i in io:
            [k, v] = i.split()[0], \
                [i.split()[3], 
                i.split()[4], 
                i.split()[6], 
                i.split()[7]
                ]
            io_dict[k] = v
        with open(self.raw_data_resource + r'/client0_memory', 'r') as f:
            mem = f.readlines()
        mem_dict = {}
        for i in mem:
            [k, v] = i.split()[0], \
                [i.split()[5], 
                i.split()[6], 
                i.split()[7]
                ]
            mem_dict[k] = v
        data_final = {}
        
        for i in io_dict:
            try:
                tmp = []
                tmp.append(cpu_dict[i])
                tmp.extend(io_dict[i])
                tmp.extend(mem_dict[i])
                tmp.append(sess_dict[i])
                data_final[i] = tmp
            except Exception as e:
                print(e)
        return data_final

    def save_csv(self, data_merge):
        df = pd.DataFrame(data = data_merge, 
            columns=['time', 'cpu_usage%', 'io_stps', 
            'io_wtps', 'io_bread/s', 'io_bwrtn/s',
            'mem_used%', 'mem_kbbuffers', 'mem_kbcached', 
            'session_num'])
        for k, v in sorted(data_merge.items()):
            time = k
            [cpu, io_stps, io_wtps, io_bread, io_bwrtn, mem_used, 
            mem_kbbuffers, mem_kbcached, session_num] = v
            df = df.append(pd.DataFrame({'time':[time],
                                        'cpu_usage%':[float(cpu)],
                                        'io_stps':[float(io_stps)],
                                        'io_wtps':[float(io_wtps)],
                                        'io_bread/s':[float(io_bread)], 
                                        'io_bwrtn/s':[float(io_bwrtn)],
                                        'mem_used%':[float(mem_used)], 
                                        'mem_kbbuffers':[float(mem_kbbuffers)], 
                                        'mem_kbcached':[float(mem_kbcached)], 
                                        'session_num':[float(session_num)]},
                                        columns=['time', 'cpu_usage%', 'io_stps', 
                                        'io_wtps', 'io_bread/s', 'io_bwrtn/s',
                                        'mem_used%', 'mem_kbbuffers', 'mem_kbcached', 
                                        'session_num']), 
                                        ignore_index=True)
        self.raw_df = df.copy()
        #pp = np.array(df.loc[:, 'cpu_usage%':'session_num'])
        #df.loc[:, 'cpu_usage%':'session_num'] = (pp - np.mean(pp, axis = 0)) / np.std(pp, axis= 0 )
        self.proceed_csv = self.proceed_path + '/' + sorted(data_merge.keys())[0] + '.csv'
        df.to_csv(self.proceed_csv , sep = ",", index = False)

    def createFolds(self):
        path_list = [
            self.raw_data_app,
            self.raw_data_resource,
            self.savedir,
            self.fill_path,
            self.proceed_path,
            self.align
        ]
        for p in path_list:
            try:
                os.mkdir(p)
            except Exception as e:
                pass

    def merge(self, sess_dict, startpoint, interval):
        #session的绝对数值转成相对增长
        sess_dict_c = sess_dict.copy()

        for i in sess_dict_c:
            if self.datetimeSub(i, startpoint) % interval == 0:
                for j in range(1, interval):
                    try:
                        sess_dict[i] = (float(sess_dict[self.strDateTime(i, sec = interval - j)]) \
                            - float(sess_dict[i])) / (interval - j + 1)
                        if sess_dict[i] < 0:
                            sess_dict[i] = 0
                        else:
                            sess_dict[i] = round(sess_dict[i], 2)
                        break
                    
                    except KeyError:
                        pass
                    
        return sess_dict


    def timeAlign(self, fpth):
        #session原本的时间减12小时是count的时间
        with open(fpth, 'r') as f:
            fd = f.readlines()
        tmp = []
        for i in fd:
            time_str, count = i.split(' ')[0], i.split(' ')[1:]
            if self.IsAM == False:
                time_add = self.strDateTime(time_str, add = 0, hour=0)
            else:
                time_add = self.strDateTime(time_str, add = 0, hour=12)
            
            
            tmp.append(time_add + ' ' + ' '.join(count))
        with open(self.align + '/' + os.path.basename(fpth).split('.')[0] + '-align.txt', 'w') as f:
            f.writelines(tmp)
        return self.align + '/' + os.path.basename(fpth).split('.')[0] + '-align.txt'

    def strDateTime(self, strTime, add = 1, sec = 0, minu = 0, hour = 0):
        if add == 1:
            [hour_, minu_, sec_] = strTime.split(':')
            hour_, minu_, sec_ = int(hour_), int(minu_), int(sec_)
            sec_ += sec
            minu_+= minu
            hour_ += hour
            if sec_ >= 60:
                sec_ -= 60
                minu_ += 1
            if minu_ >= 60:
                minu_ -= 60
                hour_ += 1
            if hour_ >= 24:
                hour_ -= 24
        else:
            [hour_, minu_, sec_] = strTime.split(':')
            hour_, minu_, sec_ = int(hour_), int(minu_), int(sec_)
            sec_ -= sec
            minu_-= minu
            hour_ -= hour
            if sec_ < 0:
                sec_ += 60
                minu_ -= 1
            if minu_ < 0:
                minu_ += 60
                hour_ -= 1
            if hour_ < 0:
                hour_ += 24
        return str(hour_).zfill(2) + ':' + str(minu_).zfill(2) + ':' + str(sec_).zfill(2)

    def datetimeSub(self, a, b):
        [h, m, s] = a.split(':')
        a_s = int(h) * 3600 + int(m) * 60 + int(s)
        [h, m, s] = b.split(':')
        b_s = int(h) * 3600 + int(m) * 60 + int(s)
        return a_s - b_s

if __name__ == "__main__":
    config_yaml = r'/home/wxh/lstmDnn/configure/project/config.yaml'
    data_set_proc = data_regular(config_yaml)
    data_set_proc.txt2csv(config_yaml)