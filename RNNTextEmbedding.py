import tensorflow as tf
from tensorflow.contrib import layers


class RNNTextEmbedding(object):
    def __init__(self, word_dim, wv, batch_size=50, hidden_size=200, layers=1, dp=0.8, max_grad_norm=5.0, lr=0.01):
        self._hidden_size = hidden_size
        self._layers = layers
        self._wv_dim = word_dim
        self._dp = dp
        self._wv = wv
        self._max_grad_norm = max_grad_norm
        self._vocab_size = len(wv)
        self._lr = lr
        self._batch_size = batch_size
        self._build_net()

    def _build_net(self):
        dp = tf.placeholder(dtype=tf.float64, shape=[])
        setattr(self, '_dropout_keep', dp)

        with tf.variable_scope('embedding'):
            # 可以考虑对x和wordvec进行dropout
            x = tf.placeholder(name='x', shape=[None, None], dtype=tf.int64)
            x_shape = tf.shape(x)
            cur_batch, cur_len = x_shape[0], x_shape[1]

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
            fcl_weights = tf.get_variable('fcl_weight', shape=[self._hidden_size, self._vocab_size], dtype=tf.float64)
            fcl_bias = tf.get_variable('fcl_bias', shape=[self._vocab_size], dtype=tf.float64)
            logits = tf.matmul(outputs, fcl_weights) + fcl_bias
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(x, [-1]), logits=logits)
            mask = tf.sequence_mask(x_size, maxlen=cur_len, dtype=tf.float64)
            mask = tf.reshape(mask, shape=[-1])
            cost = tf.reduce_sum(loss*mask) / tf.reduce_sum(mask)
            setattr(self, '_cost', cost)

            idx = tf.nn.softmax(logits)
            res_index = tf.argmax(idx, axis=1)
            res_index = tf.reshape(res_index, shape=(-1, cur_len))
            setattr(self, '_res_index', res_index)

        with tf.variable_scope('train'):
            #global_steps = tf.Variable(dtype=tf.int32, initial_value=0, trainable=False)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self._lr)
            grads, trainable_variables = zip(*optimizer.compute_gradients(cost/tf.cast(cur_batch, dtype=tf.float64)))
            grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
            setattr(self, '_train_op', train_op)

    def make_nonshuffle_dataset(self, data_path, batch_size, sep=' '):
        dataset = tf.data.TextLineDataset(data_path)
        def _parse_line(line):
            ws = tf.string_split([line], delimiter=sep).values
            ws = tf.string_to_number(ws, tf.int32)
            size = tf.size(ws)
            #ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            return ws, size

        dataset = dataset.map(_parse_line)
        dataset = dataset.filter(lambda x, size: tf.logical_and(tf.less_equal(size, 100), tf.greater_equal(size, 20)))
        padded_shapes = (tf.TensorShape([None]), tf.TensorShape([]))
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        return dataset

    def make_dataset(self, data_path, sep=' '):
        dataset = self.make_nonshuffle_dataset(data_path, self._batch_size, sep)
        return dataset.shuffle(1000)

    def fit(self, path):
        dataset = self.make_dataset(path)
        iterator = dataset.make_initializable_iterator()
        x_op, size_op = iterator.get_next()

        dp = getattr(self, '_dropout_keep')
        x = getattr(self, '_x')
        x_size = getattr(self, '_x_size')
        x_emb = getattr(self, '_x_emb')
        cost = getattr(self, '_cost')
        res_index = getattr(self, '_res_index')

        train_op = getattr(self, '_train_op')

        steps = 0
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in range(5):
                sess.run(iterator.initializer)
                while True:
                    try:
                        x_batch, size = sess.run([x_op, size_op])
                    except tf.errors.OutOfRangeError:
                        break
                    feed = {dp: self._dp, x: x_batch, x_size: size, x_emb: self._wv}
                    cur_cost, cur_index, _ = sess.run([cost, res_index, train_op], feed_dict=feed)
                    steps += 1
                    yield steps, sess, x_batch, cur_cost, cur_index


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('./word_vecs.txt', binary=False)
    wv = model.vectors
    nn = RNNTextEmbedding(50, wv, batch_size=10, hidden_size=50)
    for steps, sess, x, cost, index in nn.fit('./index.txt'):
        xs = x[:3].tolist()
        res = index[:3]
        for x, y in zip(xs, res):
            print('input: %s' % ' '.join([model.index2word[i] for i in x]))
            print('output: %s' % ' '.join([model.index2word[i] for i in y]))