#import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import sys
sys.path.append(r'/home/wxh/lstmDnn/anomaly/safekit')
import safekit.tf_ops as ops
import glob
import pandas as pd
import random
import time
import os

def lm_rnn(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level LSTM language model that uses a sentence level context vector.

    :param x: (tensor) Input to rnn
    :param t: (tensor) Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tuple) token_losses (tensor), hidden_states (list of tensors), final_hidden (tensor)
    """
    token_set_size = token_embed.get_shape().as_list()[0]
    cells = [cell(num_units) for num_units in layers]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    # mb X sentence_length X embedding_size
    #x_lookup = tf.nn.embedding_lookup(token_embed, x)
    #x_lookup = tf.expand_dims(x, axis=2)
    x_lookup = x
    
    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    # input_features: list max_length of sentence long tensors (mb X embedding_size+context_size)
    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]

    # hidden_states: sentence length long list of tensors (mb X final_layer_size)
    # cell_state: data structure that contains the cell state for each hidden layer for a mini-batch (complicated)
    hidden_states, cell_state = tf.nn.static_rnn(cell, input_features,
                                          initial_state=None,
                                          dtype=tf.float32,
                                          sequence_length=seq_len,
                                          scope='language_model')
    # batch_size X sequence_length (see tf_ops for def)
    hid_list = []
    for i in hidden_states:
        hid_list.append(tf.keras.layers.BatchNormalization()(i))
    hidden_states = hid_list
    #token_losses = ops.batch_softmax_dist_loss(t, hidden_states, token_set_size)
    token_losses = eyed_mvn_loss(t, hidden_states)
    final_hidden = cell_state[-1].h
    final_hidden = tf.keras.layers.BatchNormalization()(final_hidden)
    return token_losses, hidden_states, final_hidden

def bidir_lm_rnn(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level bidirectional LSTM language model that uses a sentence level context vector.

    :param x: Input to rnn
    :param t: Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tensor) tuple-token_losses , (list of tensors) hidden_states, (tensor) final_hidden
    """

    token_set_size = token_embed.get_shape().as_list()[0]
    with tf.variable_scope('forward'):
        fw_cells = [cell(num_units) for num_units in layers]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
    with tf.variable_scope('backward'):
        bw_cells = [cell(num_units) for num_units in layers]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)
    #x_lookup = tf.nn.embedding_lookup(token_embed, x)
    x_lookup = x

    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]
    # input_features: list of sentence long tensors (mb X embedding_size)
    hidden_states, fw_cell_state, bw_cell_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, input_features,
                                                                          dtype=tf.float32,
                                                                          sequence_length=seq_len,
                                                                          scope='language_model')
    final_hidden = tf.concat((fw_cell_state[-1].h, bw_cell_state[-1].h), 1)
    
    f_hidden_states, b_hidden_states = tf.split(tf.stack(hidden_states), 2, axis=2) # 2  sen_len X num_users X hidden_size tensors
    # truncate forward and backward output to align for prediction
    f_hidden_states = tf.stack(tf.unstack(f_hidden_states)[:-2]) # sen_len-2 X num_users X hidden_size tensor
    b_hidden_states = tf.stack(tf.unstack(b_hidden_states)[2:]) # sen_len-2 X num_users X hidden_size tensor
    # concatenate forward and backward output for prediction
    prediction_states = tf.unstack(tf.concat((f_hidden_states, b_hidden_states), 2))  # sen_len-2 long list of num_users X 2*hidden_size tensors
    #token_losses = batch_softmax_dist_loss(t, prediction_states, token_set_size)
    token_losses = eyed_mvn_loss(t, prediction_states, token_set_size)

    return token_losses, hidden_states, final_hidden

def bidir_lm_rnn_predict(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level bidirectional LSTM language model that uses a sentence level context vector.

    :param x: Input to rnn
    :param t: Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tensor) tuple-token_losses , (list of tensors) hidden_states, (tensor) final_hidden
    """

    token_set_size = token_embed.get_shape().as_list()[0]
    with tf.variable_scope('forward'):
        fw_cells = [cell(num_units) for num_units in layers]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
    with tf.variable_scope('backward'):
        bw_cells = [cell(num_units) for num_units in layers]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)
    #x_lookup = tf.nn.embedding_lookup(token_embed, x)
    x_lookup = x

    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]
    # input_features: list of sentence long tensors (mb X embedding_size)
    hidden_states, fw_cell_state, bw_cell_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, input_features,
                                                                          dtype=tf.float32,
                                                                          sequence_length=seq_len,
                                                                          #scope='language_model')
                                                                          scope=None)
    final_hidden = tf.concat((fw_cell_state[-1].h, bw_cell_state[-1].h), 1)
    
    f_hidden_states, b_hidden_states = tf.split(tf.stack(hidden_states), 2, axis=2) # 2  sen_len X num_users X hidden_size tensors
    # truncate forward and backward output to align for prediction
    #f_hidden_states = tf.stack(tf.unstack(f_hidden_states)[:-2]) # sen_len-2 X num_users X hidden_size tensor
    #b_hidden_states = tf.stack(tf.unstack(b_hidden_states)[2:]) # sen_len-2 X num_users X hidden_size tensor
    f_hidden_states = tf.stack(tf.unstack(f_hidden_states)) # sen_len-2 X num_users X hidden_size tensor
    b_hidden_states = tf.stack(tf.unstack(b_hidden_states))
    # concatenate forward and backward output for prediction
    prediction_states = tf.unstack(tf.concat((f_hidden_states, b_hidden_states), 2))  # sen_len-2 long list of num_users X 2*hidden_size tensors
    #token_losses = batch_softmax_dist_loss(t, prediction_states, token_set_size)
    token_losses = eyed_mvn_loss(t, prediction_states, token_set_size)

    return token_losses, hidden_states, final_hidden

def eyed_mvn_loss(truth, h, scale_range=1.0):
    """
    This function takes the output of a neural network after it's last activation, performs an affine transform,
    and returns the squared error of this result and the target.

    :param truth: A tensor of target vectors.
    :param h: The output of a neural network post activation.
    :param scale_range: For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for
    tanh activation and sqrt(2/fan_in) for relu activation.
    :return: (tf.Tensor[MB X D], None) squared_error, None
    """
    fan_in = h[0].get_shape().as_list()[1]
    dim = truth.shape.as_list()[2]
    U = tf.Variable(ops.fan_scale(scale_range, tf.tanh, h[0]) * tf.truncated_normal([fan_in, dim],
                                                                             dtype=tf.float32, name='U'))
    
    hidden_tensor = tf.stack(h)
    b = tf.Variable(tf.zeros([dim]))
    ustack = tf.stack([U]*len(h))
    y = tf.matmul(hidden_tensor, ustack) + b
    y = tf.transpose(y, perm=[1,0,2])
    loss_columns = tf.square(y-truth)
    #l = tf.Print(hidden_tensor,['loss_columns: ...........................',hidden_tensor])
    #hidden_tensor = l

    return loss_columns

    #cell=tf.nn.rnn_cell.BasicLSTMCell
def bidir_lm_rnn_predict_pred(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level bidirectional LSTM language model that uses a sentence level context vector.

    :param x: Input to rnn
    :param t: Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tensor) tuple-token_losses , (list of tensors) hidden_states, (tensor) final_hidden
    """
    if x.shape[0] < 70:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell(layers[0]), output_keep_prob=0.5)
    else:
        cell = cell(layers[0])

    token_set_size = token_embed.get_shape().as_list()[0]
    with tf.variable_scope('forward'):
        
        #fw_cells = [cell(num_units) for num_units in layers]
        fw_cells = [cell]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
    with tf.variable_scope('backward'):
        #bw_cells = [cell(num_units) for num_units in layers]
        bw_cells = [cell]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)
    #x_lookup = tf.nn.embedding_lookup(token_embed, x)
    x_lookup = x
    if x.shape[0] < 70:
        x_lookup = tf.nn.dropout(x_lookup, 0.5)
    

    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]
    # input_features: list of sentence long tensors (mb X embedding_size)
    hidden_states, fw_cell_state, bw_cell_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, input_features,
                                                                          dtype=tf.float32,
                                                                          sequence_length=seq_len,
                                                                          #scope='language_model')
                                                                          scope=None)
    final_hidden = tf.concat((fw_cell_state[-1].h, bw_cell_state[-1].h), 1)
    
    f_hidden_states, b_hidden_states = tf.split(tf.stack(hidden_states), 2, axis=2) # 2  sen_len X num_users X hidden_size tensors
    # truncate forward and backward output to align for prediction
    #f_hidden_states = tf.stack(tf.unstack(f_hidden_states)[:-2]) # sen_len-2 X num_users X hidden_size tensor
    #b_hidden_states = tf.stack(tf.unstack(b_hidden_states)[2:]) # sen_len-2 X num_users X hidden_size tensor
    f_hidden_states = tf.stack(tf.unstack(f_hidden_states)) # sen_len-2 X num_users X hidden_size tensor
    b_hidden_states = tf.stack(tf.unstack(b_hidden_states))
    # concatenate forward and backward output for prediction
    prediction_states = tf.unstack(tf.concat((f_hidden_states, b_hidden_states), 2))  # sen_len-2 long list of num_users X 2*hidden_size tensors
    #token_losses = batch_softmax_dist_loss(t, prediction_states, token_set_size)
    token_losses, y = eyed_mvn_loss_pred(t, prediction_states, token_set_size)

    return token_losses, hidden_states, final_hidden, y

def eyed_mvn_loss_pred(truth, h, scale_range=1.0):
    """
    This function takes the output of a neural network after it's last activation, performs an affine transform,
    and returns the squared error of this result and the target.

    :param truth: A tensor of target vectors.
    :param h: The output of a neural network post activation.
    :param scale_range: For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for
    tanh activation and sqrt(2/fan_in) for relu activation.
    :return: (tf.Tensor[MB X D], None) squared_error, None
    """
    fan_in = h[0].get_shape().as_list()[1]
    dim = truth.shape.as_list()[2]
    #with tf.variable_scope('outpt',reuse=True): 
    U = tf.Variable(ops.fan_scale(scale_range, tf.tanh, h[0]) * tf.truncated_normal([fan_in, dim],
                                                                            dtype=tf.float32, name='U'))
    b = tf.Variable(tf.zeros([dim]))

    hidden_tensor = tf.stack(h)
    
    

    ustack = tf.stack([U]*len(h))
    y = tf.matmul(hidden_tensor, ustack) + b
    y = tf.transpose(y, perm=[1,0,2])
    loss_columns = tf.square(y-truth)
    
    #l = tf.Print(hidden_tensor,['loss_columns: ...........................',hidden_tensor])
    #hidden_tensor = l
    
    
    return loss_columns, y


class OnlineBatcher:
    """
    Gives batches from a csv file.
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, batch_size,
                 skipheader=False, delimiter=',',
                 alpha=0.5, size_check=None,
                 datastart_index=3, norm=False):
        """

        :param datafile: (str) File to read lines from.
        :param batch_size: (int) Mini-batch size.
        :param skipheader: (bool) Whether or not to skip first line of file.
        :param delimiter: (str) Delimiter of csv file.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param size_check: (int) Expected number of fields from csv file. Used to check for data corruption.
        :param datastart_index: (int) The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        :param norm: (bool) Whether or not to normalize the real valued data features.
        """

        self.alpha = alpha
        self.f = open(datafile, 'r')
        self.batch_size = batch_size
        self.index = 0
        self.delimiter = delimiter
        self.size_check = size_check
        if skipheader:
            self.header = self.f.readline()
        self.datastart_index = datastart_index
        self.norm = norm
        self.replay = False

    def next_batch(self):
        """
        :return: (np.array) until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        matlist = []
        l = self.f.readline()
        lsplit = l.strip().split(self.delimiter)
        if l == '':
            return None
        l = lsplit[0].replace(':','') + ',' + ','.join(lsplit[1:])
        rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        if self.size_check is not None:
            while len(rowtext) != self.size_check:
                l = self.f.readline()
                lsplit = l.strip().split(self.delimiter)
                if l == '':
                    return None
                l = lsplit[0].replace(':','') + ',' + ','.join(lsplit[1:])
                rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
        matlist.append(rowtext)
        for i in range(self.batch_size - 1):
            l = self.f.readline()
            lsplit = l.strip().split(self.delimiter)
            if l == '':
                break
            l = lsplit[0].replace(':','') + ',' + ','.join(lsplit[1:])
            rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            if self.size_check is not None:
                while len(rowtext) != self.size_check:
                    l = self.f.readline()
                    lsplit = l.strip().split(self.delimiter)
                    if l == '':
                        return None
                    l = lsplit[0].replace(':','') + ',' + ','.join(lsplit[1:])
                    rowtext = np.array([float(k) for k in l.strip().split(self.delimiter)])
            matlist.append(rowtext)
        data = np.array(matlist)
        if self.norm:
            batchmean, batchvariance = data[:,self.datastart_index:].mean(axis=0), data[:, self.datastart_index:].var(axis=0)
            if self.index == 0:
                self.mean, self.variance = batchmean, batchvariance
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
                self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
                data[:, self.datastart_index:] = (data[:, self.datastart_index:] - self.mean)/(self.variance + 1e-10)
        self.index += self.batch_size
        return data


class OnlineBatcher_custom:
    """
    Gives batches from a csv file.
    For batching data too large to fit into memory. Written for one pass on data!!!
    """

    def __init__(self, datafile, batch_size, sentence, 
                 datafeature, train,resource,
                 anomaly=False,big_data=False,
                 skipheader=False, delimiter=',',
                 alpha=0.5, size_check=None,
                 norm=False):
        """

        :param datafile: (str) File to read lines from.
        :param batch_size: (int) Mini-batch size.
        :param skipheader: (bool) Whether or not to skip first line of file.
        :param delimiter: (str) Delimiter of csv file.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param size_check: (int) Expected number of fields from csv file. Used to check for data corruption.
        :param datastart_index: (int) The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        :param norm: (bool) Whether or not to normalize the real valued data features.
        """
        random.seed(300)
        self.sentence = sentence
        self.alpha = alpha
        self.anomaly = anomaly
        self.files = glob.glob(datafile + '/*/*.csv')
        if big_data == False:
            self.trainFile = random.sample(self.files, int(len(self.files) * 0.8))
        else:
            self.trainFile = random.choice(self.files, int(len(self.files) * 0.8), replace = False)
        if train == False:
            self.trainFile = list(set(self.files) - set(self.trainFile))
        self.datafeature = datafeature
        self.batch_size = batch_size
        self.index = 0
        self.delimiter = delimiter
        self.size_check = size_check
        self.datastart_index = self.datafeature[5]
        self.norm = norm
        self.replay = False
        self.fileFlag = 0
        self.sentence_flag = 0
        self.resource = resource

    def next_batch(self):
        """
        :return: (np.array) until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        readSize = 0
        f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
        dtt = f.copy()
        dtt.iloc[:, 5] = f.iloc[:, 5] * f.iloc[0, -3]
        dtt.iloc[:, [6, 8, 9]] = f.iloc[:, [6, 8, 9]] * f.iloc[0, -2]
        last_dim = f.shape[-1]
        data = np.empty([self.batch_size, self.sentence, last_dim])
        f_len = len(f)
        while readSize != self.batch_size:
            if self.anomaly == False:
                f_read = f.iloc[self.sentence_flag:self.sentence_flag + self.sentence, :]
                if np.sum(np.array(dtt.iloc[:, 6] >= self.resource['cpu'])) == 0 and len(f_read)==self.sentence:
                    a_1 = np.array(f_read.iloc[:-1, 1])
                    a_2 = np.array(f_read.iloc[1:, 1])
                    inconsistance = np.where(a_2 - a_1 >= 300000000 * 2, 1, 0)
                    if np.sum(inconsistance) == 0 and np.any(f_read.isnull()) == 0:
                        try:
                            data[readSize, :, :] = f_read
                        except Exception as e:
                            print(inconsistance)
                            print(e)
                            print(self.trainFile[self.fileFlag])
                            
                            sys.exit(0)
                        readSize += 1
                    self.sentence_flag += 1
                    if self.sentence_flag == (f_len - self.sentence - 2):
                        self.fileFlag += 1
                        if self.fileFlag == len(self.trainFile):
                            return None
                        f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                        self.sentence_flag = 0
                        f_len = len(f)
                else:
                    self.fileFlag += 1
                    if self.fileFlag == len(self.trainFile):
                        return None
                    f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                    f_len = len(f)
                    self.sentence_flag = 0


            else:
                f_read = f.iloc[self.sentence_flag:self.sentence_flag + self.sentence, :]
                if np.sum(np.array(dtt.iloc[:, 6] >= self.resource['cpu'])) != 0 and len(f_read)==self.sentence:
                    
                    a_1 = np.array(f_read.iloc[:-1, 1])
                    a_2 = np.array(f_read.iloc[1:, 1])
                    inconsistance = np.where(a_2 - a_1 >= 300000000 * 2, 1, 0)
                    if np.sum(inconsistance) == 0 and np.any(f_read.isnull()) == 0:
                        
                        data[readSize, :, :] = f_read
                        readSize += 1
                    self.sentence_flag += 1
                    if self.sentence_flag == (f_len - self.sentence - 2):
                        self.fileFlag += 1
                        if self.fileFlag == len(self.trainFile):
                            return None
                        f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                        self.sentence_flag = 0
                        f_len = len(f)
                else:
                    self.fileFlag += 1
                    if self.fileFlag == len(self.trainFile):
                            return None
                    f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                    self.sentence_flag = 0
                    f_len = len(f)

        if self.norm:
            batchmean, batchvariance = data[:, :, self.datastart_index:].mean(axis=(0, 1)), data[:, :, self.datastart_index:].std(axis=(0, 1))
            if self.index == 0:
                self.mean, self.variance = batchmean, batchvariance
            else:
                self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
                self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
            
            data[:, :, self.datastart_index:] = (data[:, :, self.datastart_index:] - self.mean)/(self.variance + 1e-10)
        self.index += self.batch_size
        return data

class OnlineBatcher_custom_mem():
    """
    Gives batches from a csv file.
    For batching data too large to fit into memory. Written for one pass on data!!!
    """
    #transform data to gpu mem 

    def __init__(self, datafile, batch_size, sentence, 
                 datafeature, train,resource,
                 anomaly=False,big_data=False,
                 skipheader=False, delimiter=',',
                 alpha=0.5, size_check=None,
                 norm=False, is_cached = True):
        """

        :param datafile: (str) File to read lines from.
        :param batch_size: (int) Mini-batch size.
        :param skipheader: (bool) Whether or not to skip first line of file.
        :param delimiter: (str) Delimiter of csv file.
        :param alpha: (float)  For exponential running mean and variance.
                      Lower alpha discounts older observations faster.
                      The higher the alpha, the further you take into consideration the past.
        :param size_check: (int) Expected number of fields from csv file. Used to check for data corruption.
        :param datastart_index: (int) The csv field where real valued features to be normalized begins.
                                Assumed that all features beginnning at datastart_index till end of line
                                are real valued.
        :param norm: (bool) Whether or not to normalize the real valued data features.
        """
        random.seed(300)
        self.sentence = sentence
        self.alpha = alpha
        self.anomaly = anomaly
        self.files = glob.glob(datafile + '/*/*.csv')
        if big_data == False:
            self.trainFile = random.sample(self.files, int(len(self.files) * 0.8))
        else:
            self.trainFile = random.choice(self.files, int(len(self.files) * 0.8), replace = False)
        if train == False:
            self.trainFile = list(set(self.files) - set(self.trainFile))
        self.datafeature = datafeature
        self.batch_size = batch_size
        self.index = 0
        self.delimiter = delimiter
        self.size_check = size_check
        self.datastart_index = self.datafeature[5]
        self.norm = norm
        self.replay = False
        self.fileFlag = 0
        self.sentence_flag = 0
        self.resource = resource
        if train:
            #self.path = r'/home/wxh/lstmDnn/anomaly/cache/array' + \
            #            str(self.batch_size) + '_train.dat.npy'
            #self.path = r'/home/wxh/lstmDnn/anomaly/cache/array64_Normal_train.dat.npy'
            self.path = r'/home/wxh/lstmDnn/anomaly/cache/array64_train_6feature_Normal.dat.npy'
            if self.norm:
                self.path = r'/home/wxh/lstmDnn/anomaly/cache/array' + \
                            str(self.batch_size) + '_onlineBN_train_6feature_Normal.dat.npy'
        else:
            #self.path = r'/home/wxh/lstmDnn/anomaly/cache/array' + \
            #            str(self.batch_size) + '_test.dat.npy'
            #self.path = r'/home/wxh/lstmDnn/anomaly/cache/array64_Normal_test.dat.npy'
            self.path = r'/home/wxh/lstmDnn/anomaly/cache/array64_test_6feature_Normal.dat.npy'
            if self.norm:
                self.path = r'/home/wxh/lstmDnn/anomaly/cache/array' + \
                            str(self.batch_size) + '_onlineBN_test_6feature_test.dat.npy'
        self.saved = 0
        self.cache = []
        if is_cached:
            if os.path.exists(self.path):
                #print('load... ' + self.path)
                self.saved = 1
                self.data = np.load(self.path)
                self.len = len(self.data)
                self.counter = -1
                #self.data = tf.constant(self.data)
            

    def next_batch(self):
        """
        :return: (np.array) until end of datafile, each time called,
                 returns mini-batch number of lines from csv file
                 as a numpy array. Returns shorter than mini-batch
                 end of contents as a smaller than batch size array.
                 Returns None when no more data is available(one pass batcher!!).
        """
        if self.saved == 0:
            readSize = 0
            f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
            dtt = f.copy()
            dtt.iloc[:, 5] = f.iloc[:, 5] * f.iloc[0, -3]
            dtt.iloc[:, [6, 8, 9]] = f.iloc[:, [6, 8, 9]] * f.iloc[0, -2]
            last_dim = f.shape[-1]
            data = np.empty([self.batch_size, self.sentence, last_dim])
            f_len = len(f)
            while readSize != self.batch_size:
                if self.anomaly == False:
                    f_read = f.iloc[self.sentence_flag:self.sentence_flag + self.sentence, :]
                    if np.sum(np.array(dtt.iloc[:, 6] >= self.resource['cpu'])) == 0 and len(f_read)==self.sentence:
                        a_1 = np.array(f_read.iloc[:-1, 1])
                        a_2 = np.array(f_read.iloc[1:, 1])
                        inconsistance = np.where(a_2 - a_1 >= 300000000 * 2, 1, 0)
                        if np.sum(inconsistance) == 0 and np.any(f_read.isnull()) == 0:
                            try:
                                data[readSize, :, :] = f_read
                            except Exception as e:
                                print(inconsistance)
                                print(e)
                                print(self.trainFile[self.fileFlag])
                                
                                sys.exit(0)
                            readSize += 1
                        self.sentence_flag += 1
                        if self.sentence_flag == (f_len - self.sentence - 2):
                            self.fileFlag += 1
                            if self.fileFlag == len(self.trainFile):
                                np.save(self.path, np.array(self.cache).reshape(-1, 
                                        self.cache[0].shape[0], self.cache[0].shape[1],
                                        self.cache[0].shape[2]))
                                return None
                            f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                            self.sentence_flag = 0
                            f_len = len(f)
                    else:
                        self.fileFlag += 1
                        if self.fileFlag == len(self.trainFile):
                            np.save(self.path, np.array(self.cache).reshape(-1, 
                                        self.cache[0].shape[0], self.cache[0].shape[1],
                                        self.cache[0].shape[2]))
                            return None
                        f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                        f_len = len(f)
                        self.sentence_flag = 0


                else:
                    f_read = f.iloc[self.sentence_flag:self.sentence_flag + self.sentence, :]
                    if np.sum(np.array(dtt.iloc[:, 6] >= self.resource['cpu'])) != 0 and len(f_read)==self.sentence:
                        
                        a_1 = np.array(f_read.iloc[:-1, 1])
                        a_2 = np.array(f_read.iloc[1:, 1])
                        inconsistance = np.where(a_2 - a_1 >= 300000000 * 2, 1, 0)
                        if np.sum(inconsistance) == 0 and np.any(f_read.isnull()) == 0:
                            
                            data[readSize, :, :] = f_read
                            readSize += 1
                        self.sentence_flag += 1
                        if self.sentence_flag == (f_len - self.sentence - 2):
                            self.fileFlag += 1
                            if self.fileFlag == len(self.trainFile):
                                np.save(self.path, np.array(self.cache).reshape(-1, 
                                        self.cache[0].shape[0], self.cache[0].shape[1],
                                        self.cache[0].shape[2]))
                                return None
                            f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                            self.sentence_flag = 0
                            f_len = len(f)
                    else:
                        self.fileFlag += 1
                        if self.fileFlag == len(self.trainFile):
                            np.save(self.path, np.array(self.cache).reshape(-1, 
                                        self.cache[0].shape[0], self.cache[0].shape[1],
                                        self.cache[0].shape[2]))
                            return None
                        f = pd.read_csv(self.trainFile[self.fileFlag], header = None).iloc[:, self.datafeature]
                        self.sentence_flag = 0
                        f_len = len(f)

            if self.norm:
                batchmean, batchvariance = data[:, :, self.datastart_index:].mean(axis=(0, 1)), data[:, :, self.datastart_index:].std(axis=(0, 1))
                if self.index == 0:
                    self.mean, self.variance = batchmean, batchvariance
                else:
                    self.mean = self.alpha * self.mean + (1 - self.alpha) * batchmean
                    self.variance = self.alpha * self.variance + (1 - self.alpha) * batchvariance
                
                data[:, :, self.datastart_index:] = (data[:, :, self.datastart_index:] - self.mean)/(self.variance + 1e-10)

            self.index += self.batch_size
            self.cache.append(data)
            return data
        else:
            self.counter += 1
            if self.counter != self.len:
                return self.data[self.counter]
                
                '''data_slice = tf.slice(self.data, [self.counter, 0, 0, 0], 
                                    [1, self.data.shape[1], self.data.shape[2],
                                    self.data.shape[3]])
                return tf.squeeze(data_slice)'''
            else:
                return None




class ArtificialDataset(tf.data.Dataset):


    def _generator(f, mb_size, 
                    sentence, datafeature,
                    is_training,  
                    anomaly, delimiter=',', 
                    skipheader=False, norm=False):
        #spawn all batch data.
        dataloader = OnlineBatcher_custom(f.decode(), mb_size, sentence, 
                                            datafeature, is_training, 
                                            resource={'cpu':0.9},anomaly=anomaly,  delimiter=delimiter, 
                                            skipheader=skipheader, norm = norm)
        # Opening the file
        idx = 0
        while 1:
            data = dataloader.next_batch()
            yield data
            idx += 1
            if data is None:
                break
                if idx == 10:
                    break

        '''for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            time.sleep(0.015)

            yield (sample_idx,)'''

    def __new__(cls,f, mb_size, 
                    sentence, datafeature,
                    is_training,  
                    anomaly, delimiter=',', 
                    skipheader=False, norm=False):
        '''return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.int64),
            args=(num_samples,)'''
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (64,8,15), dtype = tf.float32),
            args=(f, mb_size, sentence, datafeature,
            is_training, 
            anomaly, delimiter, 
            skipheader, norm,)
        )

def benchmark(dataset, num_epochs=50):
    start_time = time.perf_counter()
    dataset = dataset.make_one_shot_iterator()
    
    for epoch_num in range(num_epochs):
        #for sample in dataset:
        dataset.get_next()
        # Performing a training step
        time.sleep(0.01)
        pass
    print("Execution time:", time.perf_counter() - start_time)


def npy2normal(pth):
    data = np.load(pth)
    mean, std = np.mean(np.unique(data, axis=2),axis=(0,1,2)), np.std(np.unique(data, axis=2),axis=(0,1,2))
    data = (data - mean) / std
    p_1, p_2 = pth.split('_')
    np.save(p_1 + '_Normal_' + p_2, data)
    


if __name__ == '__main__':
    pth = r'/home/wxh/lstmDnn/anomaly/cache/array64_train.dat.npy'
    npy2normal(pth)