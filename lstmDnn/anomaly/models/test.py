import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()
import time
import numpy as np


def processData(npy):
    data = np.load(npy)
    endx = data.shape[1] - 1
    endt = data.shape[1]
    datax = data[:, :endx, 5:-3],
    datat = data[:, 1:endt, 5:-3]
    data = 
