import tensorflow as tf 
import pdb
import numpy as np
class config:
  def __init__(self):
    self.in_width = 33
    self.in_height = 8
    self.in_depth = 9

    self.seq_len = 5

    self.batch_size = 5
    self.kernel_size = 3
    self.rnn_hidden_size = 128
    self.rnn_layers = 1
    
    self.learning_rate = 0.005
    self.epochs = 10

class Model:
  def __init__(self):
    self.config = config()
    # input tensor
    self.x = tf.placeholder("float", shape=[None, self.config.seq_len, self.config.in_height, self.config.in_width, self.config.in_depth])
    self.y = tf.placeholder("float", shape=[None, self.config.seq_len, self.config.in_height, self.config.in_width, self.config.in_depth])
    # conv layers
    self.conv_output = self._get_conv_layers()
    # rnn layers
    self.rnn_output = self._get_rnn_layers()
    # pdb.set_trace()
    # de-conv layers
    self.x_hat = self._get_decoder_layers()
    # L2 Loss
    self.loss = tf.reduce_mean(tf.squared_difference(self.x_hat, self.y))
    # optimization
    self.train_step = self._adam_optimize()
    # initialization
    self.init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(self.init_op)

  def fit(self, X, y, X_val=None, y_val=None):
    for i in range(self.config.epochs):
      idx = np.arange(X.shape[0])
      np.random.shuffle(idx)
      train_loss = []
      start = 0
      # print('Epoch: %6s / %6s' % (i, self.config.epochs))
      while start < X.shape[0]:
        batch_idx = idx[start:start + self.config.batch_size]
        result, _ = self.sess.run([self.loss, self.train_step], feed_dict={self.x:X[batch_idx], self.y:y[batch_idx]})
        train_loss.append(result)
        start += self.config.batch_size
      tr_loss = sum(train_loss)/len(train_loss)
      te_loss = self.sess.run([self.loss], feed_dict={self.x:X_val, self.y:y_val})[0]
      print('Epoch: %6s / %6s : training loss = %5.3f, validation loss = %5.3f' %(i, self.config.epochs, tr_loss, te_loss))



  def predict(self, X):
    result = self.sess.run([self.x_hat], feed_dict={self.x:X})
    return result

  def _get_conv_layers(self):
    batch_size = tf.shape(self.x)[0]
    x_reshape = tf.reshape(self.x, shape=[batch_size*self.config.seq_len, self.config.in_height, self.config.in_width, self.config.in_depth])
    conv_1 = tf.layers.conv2d(x_reshape, filters=4, kernel_size=self.config.kernel_size, padding='same')
    # pdb.set_trace()
    conv_2 = tf.layers.conv2d(conv_1, filters=4, kernel_size=self.config.kernel_size, padding='same')
    conv_output = tf.reshape(conv_2, shape=tf.shape(self.x))
    conv_flatten = tf.reshape(conv_output, shape=[batch_size, self.config.seq_len, self.config.in_width*self.config.in_height*4])
    return conv_flatten

  def _get_rnn_layers(self):
    stacked_cell = [tf.contrib.rnn.LSTMCell(self.config.rnn_hidden_size) for _ in range(self.config.rnn_layers)]
    cell_enc = tf.contrib.rnn.MultiRNNCell(cells = stacked_cell, state_is_tuple = True)
    enc_inputs = tf.unstack(self.conv_output, axis = 1)
    initial_state_enc = cell_enc.zero_state(tf.shape(self.x)[0], tf.float32)
    outputs_enc,_ = tf.contrib.legacy_seq2seq.rnn_decoder(enc_inputs, initial_state_enc,cell_enc)
    outputs_enc = [tf.expand_dims(t, axis=1) for t in outputs_enc]
    rnn_out = tf.concat(outputs_enc, axis=1)
    return rnn_out


  def _get_decoder_layers(self):
    decode_out = tf.layers.dense(self.rnn_output, self.config.in_height*self.config.in_width*self.config.in_depth)
    shape = tf.shape(self.x)
    decode_out = tf.reshape(decode_out, [shape[0], shape[1],self.config.in_height, self.config.in_width, self.config.in_depth])
    return decode_out

  def _adam_optimize(self):
    global_step = tf.Variable(0,trainable=False)
    lr = tf.train.exponential_decay(self.config.learning_rate,global_step,1000,0.1,staircase=False)    
    #Route the gradients so that we can plot them on Tensorboard
    tvars = tf.trainable_variables()
    #We clip the gradients to prevent explosion
    grads = tf.gradients(self.loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads,5)
    self.numel = tf.constant([[0]])

    #And apply the gradients
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = zip(grads, tvars)
    train_step = optimizer.apply_gradients(gradients,global_step=global_step)
    return train_step


if __name__ == '__main__':

  model = Model()
  for var in tf.trainable_variables():
  	print(var)
  	print(var.shape)
  # pdb.set_trace()
  X_train = np.random.random([20, 5, 8, 33, 9])
  y_train = np.random.random([20, 5, 8, 33, 9])
  X_val = np.random.random([10, 5, 8, 33, 9])
  y_val = np.random.random([10, 5, 8, 33, 9])
  X_test = np.random.random([10, 5, 8, 33, 9])
  model.fit(X_train, y_train, X_val, y_val)
  y_pred = model.predict(X_test)
  # print(y_pred)