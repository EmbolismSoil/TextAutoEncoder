import tensorflow as tf
from tensorflow.contrib import layers


class RNNTextEmbedding(object):
    def __init__(self, word_dim, wv, hidden_size=200, layers=2, dp=0.8, max_grad_norm=5.0, lr=0.01):
        self._hidden_size = hidden_size
        self._layers = layers
        self._wv_dim = word_dim
        self._dp = dp
        self._wv = wv
        self._max_grad_norm = max_grad_norm
        self._vocab_size = 30000#len(wv)
        self._lr = lr
        self._build_net()

    def _build_net(self):

        dp = tf.placeholder(dtype=tf.float64, shape=[])
        setattr(self, '_dp', dp)

        with tf.variable_scope('embedding'):
            # 可以考虑对x和wordvec进行dropout
            x = tf.placeholder(name='x', shape=[None, None], dtype=tf.int64)
            cur_batch, cur_len = tf.shape(x)

            x_size = tf.placeholder(name='x_size', shape=[None], dtype=tf.int64)
            x_emb = tf.placeholder(name='x_emb', shape=[None, self._wv_dim], dtype=tf.float64)
            setattr(self, '_x', x)
            setattr(self, '_x_size', x_size)
            setattr(self, '_x_emb', x_emb)

            x_embedded = tf.nn.embedding_lookup(x_emb, x)

        with tf.variable_scope('encoder'):
            enc_cells = [tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size) for _ in range(self._layers)]
            encoder = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(enc_cells), output_keep_prob=dp)
            setattr(self, '_encoder', encoder)

            enc_outputs, enc_states = tf.nn.dynamic_rnn(encoder, x_embedded, x_size, dtype=tf.float64)

        with tf.variable_scope('decoder'):
            dec_cells = [tf.nn.rnn_cell.BasicLSTMCell(self._hidden_size) for _ in range(self._layers)]
            decoder = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.MultiRNNCell(dec_cells), output_keep_prob=dp)
            setattr(self, '_decoder', decoder)

            dec_outputs, _ = tf.nn.dynamic_rnn(decoder, x_embedded, x_size, initial_state=enc_states)

        with tf.variable_scope('output'):
            outputs = tf.reshape(dec_outputs, shape=(-1, self._hidden_size))
            logits = layers.fully_connected(outputs, self._vocab_size, activation_fn=None)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(x, [-1], logits=logits))
            mask = tf.sequence_mask(x_size, maxlen=cur_len, dtype=tf.float64)
            cost = tf.reduce_sum(loss*mask) / tf.reduce_sum(mask)

        with tf.variable_scope('train'):
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self._lr)
            trainable_variables = tf.trainable_variables()
            grads = optimizer.compute_gradients(cost/tf.to_float(cur_batch), trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
            train_op = optimizer.apply_gradients(grads)
            setattr(self, 'train_op', train_op)



if __name__ == '__main__':
    nn = RNNTextEmbedding(128, None)