from operator import indexOf
import pandas as pd
import glob
import os

path = r'/home/wxh/capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/resources/workload/my-trace'

cpu_cap = {'4665896876': 0.1038, 
           '6162908962': 0.1562,
           '3727144341': 0.125,
           '3769734721': 0.1943,
           '1863690462': 0.03125}
mem_cap = {'4665896876': 0.07642, 
           '6162908962': 0.1399,
           '3727144341': 0.004662,
           '3769734721': 0.1375,
           '1863690462': 0.006218}


trace = glob.glob(r'/home/wxh/capacityPlanning/simulation/manoelcampos-cloudsimplus-cc58449/cloudsim-plus-examples/src/main/resources/workload/my-trace/*.csv')
max_cpu = sorted(cpu_cap.items(),key = lambda x:x[1],reverse = True)[0][1]
max_mem = sorted(mem_cap.items(),key = lambda x:x[1],reverse = True)[0][1]
for idx, i in enumerate(trace):
    df = pd.read_csv(i)
    key = i.split('/')[-1].split('_')[0]
    df.iloc[:, 4] = df.iloc[:, 4] / max_cpu * 100
    df.iloc[:, 5] = df.iloc[:, 5] / max_mem * 100 / 20 #trace读取的ram只能是百分比，为了让程序不触发swap，手动修改ram
    mem_pth = os.path.dirname(i) + '/' + '_'.join(os.path.basename(i).split('_')[:2] + ['mem'] + [str(mem_cap[key])])
    cpu_pth = os.path.dirname(i) + '/' + '_'.join(os.path.basename(i).split('_')[:2] + ['cpu'] + [str(cpu_cap[key])])
    cpu = open(cpu_pth, 'w')
    for j in df.iloc[:, 4]:
        cpu.write(str(j) + '\n')
    mem = open(mem_pth, 'w')
    for j in df.iloc[:, 5]:
        mem.write(str(j) + '\n')
    