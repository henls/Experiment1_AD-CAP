import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def show_log(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
    x = []
    y = []

    for i in lines:
        try:
            x.append(np.float(i.split(' ')[3]))
            y.append(np.float(i.split(' ')[5]))
        except Exception as e:
            print(i)
    length = 200
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(x))[:length], x[:length], c='blue', label='train_set(cpu_load:88 std:1.7)')
    plt.plot(range(len(x))[:length], y[:length], c='red', label='valid_set(cpu_load:25 std:1.4)')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig_n = pth.split('/')[-1].split('.')[0]
    plt.legend()
    plt.savefig('/home/wxh/AnomalyDetectionModels/safekit-master/examples/' + fig_n + '.png')


if __name__ == '__main__':
    pth = r'/home/wxh/AnomalyDetectionModels/safekit-master/examples/084123-151540.log'
    show_log(pth)
