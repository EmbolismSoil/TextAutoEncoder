import tensorflow as tf
from tensorflow import layers
import numpy as np


class TextCNNAutoEncoder(object):
    def __init__(self, sentence_len, wv_mat,
                 word_vec_dim=300, batch_size=50,
                 filter_sizes=(2, 3), filter_depth=50,
                 drop_prob=0.5, lr=0.03, epochs=5, logdir=None):
        self._sentence_len = sentence_len
        self._wv_mat = wv_mat
        self._word_vec_dim = word_vec_dim
        self._batch_size = batch_size
        self._filter_sizes = filter_sizes
        self._filter_depth = filter_depth
        self._drop_prob = drop_prob
        self._lr = lr
        self._epochs = epochs
        self._logdir = logdir
        self._vocab_size = len(wv)

        self._build_net()

    def _build_net(self):
        x = tf.placeholder(tf.int64, shape=[None, self._sentence_len], name='embedding-layer-x')
        size = tf.placeholder(tf.int64, shape=(None,), name='origin_len')
        dr = tf.placeholder(tf.float64, shape=[])
        cur_batch_size = tf.shape(x)[0]
        x_emb = tf.placeholder(tf.float64, shape=[None, self._word_vec_dim], name='embedding-table')
        x_embedded = tf.nn.embedding_lookup(x_emb, x)
        x_embedded = tf.reshape(x_embedded, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])

        flatten_weight = tf.Variable(dtype=tf.float64,trainable=True,
                                     initial_value=tf.random_normal(dtype=tf.float64,
                                                                    shape=(self._word_vec_dim, self._vocab_size)))
        flatten_bias = tf.Variable(dtype=tf.float64,
                                   initial_value=tf.random_normal(dtype=tf.float64,
                                                                    shape=[self._vocab_size]))
        setattr(self, 'x', x)
        setattr(self, 'x_emb', x_emb)
        setattr(self, 'dr', dr)
        setattr(self, 'origin_len', size)

        # CNN Layers
        conv_outs = []
        masks = []
        for idx, filter_size in enumerate(self._filter_sizes):
            out = layers.conv2d(x_embedded,
                                filters=self._filter_depth,
                                kernel_size=(filter_size, self._word_vec_dim),
                                activation=tf.nn.relu, padding='VALID',
                                use_bias=True,
                                bias_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                name='conv_layer_%d' % idx)

            out = layers.dropout(out, self._drop_prob)
            features_len = self._sentence_len - filter_size + 1
            out, mask = tf.nn.max_pool_with_argmax(out, ksize=(1, features_len, 1, 1), padding='VALID', strides=[1, 1, 1, 1])
            masks.append(mask)
            conv_outs.append(out)

        concated = tf.concat(conv_outs, axis=3, name='concated')
        reshaped = tf.reshape(concated, shape=[cur_batch_size, -1], name='reshaped')
        encoded = reshaped

        #decoder
        deconv_outs = []
        for i, (mask, out, filter_size) in enumerate(zip(masks, conv_outs, self._filter_sizes)):
            mask = tf.reshape(mask, shape=[cur_batch_size, self._filter_depth])
            idx = (mask - np.asarray(range(self._filter_depth))) / self._filter_depth
            features_len = self._sentence_len - filter_size + 1
            b = tf.range(tf.cast(cur_batch_size, dtype=tf.float64), delta=1.0, dtype=tf.float64)
            b = tf.transpose(b)*features_len
            b = tf.reshape(b, shape=[-1, 1])
            idx = idx - b
            idx = tf.cast(idx, tf.int64)
            idx = tf.one_hot(idx, depth=features_len, axis=1, dtype=tf.float64)
            idx = tf.reshape(idx, shape=[cur_batch_size, features_len, 1, self._filter_depth])
            unpool = tf.multiply(idx, out, name='unpooling_%d' % i)

            deconv_out = layers.conv2d_transpose(unpool,
                                filters=self._filter_depth,
                                kernel_size=(filter_size, self._word_vec_dim),
                                activation=tf.nn.relu, padding='VALID',
                                use_bias=True,
                                bias_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                name='deconv_layer_%d' % i)
            deconv_out = layers.dropout(deconv_out, self._drop_prob)
            deconv_outs.append(deconv_out)

        res_concated = tf.concat(deconv_outs, axis=3)
        # res_sentence = tf.nn.max_pool(res_concated, ksize=(1, 1, 150, 1), padding='VALID', strides=[1, 1, 1, 1])
        res_sentence = tf.reduce_mean(res_concated, axis=3)
        res_sentence = tf.reshape(res_sentence, shape=(-1, self._word_vec_dim))
        res_sentence = tf.matmul(res_sentence, flatten_weight) + flatten_bias
        #res_sentence_reshaped = tf.reshape(res_sentence, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(x, [-1]), logits=res_sentence)
        seq_mask = tf.sequence_mask(size, maxlen=self._sentence_len, dtype=tf.float64)
        seq_mask = tf.reshape(seq_mask, shape=[-1])
        loss = tf.reduce_sum(loss*seq_mask)
        loss = loss / tf.reduce_sum(seq_mask)

        res_sentence = tf.nn.softmax(res_sentence)
        res_index = tf.argmax(res_sentence, axis=1)
        res_index = tf.reshape(res_index, shape=(-1, self._sentence_len))

        tf.summary.scalar('loss', loss)
        setattr(self, 'encoded', encoded)
        setattr(self, 'res_sentence', res_index)
        setattr(self, 'loss', loss)

    def make_nonshuffle_dataset(self, data_path, batch_size, sep=' '):
        dataset = tf.data.TextLineDataset(data_path)
        def _parse_line(line):
            ws = tf.string_split([line], delimiter=sep).values
            ws = tf.string_to_number(ws, tf.int32)
            size = tf.size(ws)
            ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            return ws, size

        dataset = dataset.map(_parse_line)
        dataset = dataset.filter(lambda x, size: tf.greater_equal(size, 50))
        padded_shapes = (tf.TensorShape([self._sentence_len]), tf.TensorShape([]))
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=(0, 0))
        return dataset

    def make_nonshuffle_predict_dataset(self, data_path, batch_size, sep=' '):
        def _parse_line(line):
            items = tf.string_split([line], delimiter=' ').values
            ws = tf.string_to_number(items, tf.int32)
            ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            return ws

        dataset = tf.data.TextLineDataset(data_path)
        dataset = dataset.map(_parse_line)
        dataset = dataset.padded_batch(batch_size, padded_shapes=[self._sentence_len],
                                       padding_values=0)
        return dataset

    def make_dataset(self, data_path, sep=' '):
        dataset = self.make_nonshuffle_dataset(data_path, self._batch_size, sep)
        return dataset.shuffle(1000)

    def fit(self, path):
        dataset = self.make_dataset(path)
        iterator = dataset.make_initializable_iterator()
        x_op, size_op = iterator.get_next()

        with tf.Session() as sess:
            x = getattr(self, 'x')
            x_emb = getattr(self, 'x_emb')
            dr = getattr(self, 'dr')
            loss = getattr(self, 'loss')
            res_sentence = getattr(self, 'res_sentence')
            origin_size = getattr(self, 'origin_len')

            global_steps = tf.Variable(dtype=tf.int32, initial_value=0, trainable=False)
            train_op = tf.train.AdamOptimizer(self._lr).minimize(loss, global_step=global_steps)

            merged = None
            writer = None
            if self._logdir is not None:
                merged = tf.summary.merge_all()
                writer = tf.summary.FileWriter(self._logdir, sess.graph)

            sess.run(tf.initialize_all_variables())
            for i in range(self._epochs):
                sess.run(iterator.initializer)
                while True:
                    try:
                        x_batch, size = sess.run([x_op, size_op])
                    except tf.errors.OutOfRangeError:
                        break
                    feed = {x: x_batch, x_emb: self._wv_mat, dr: self._drop_prob, origin_size: size}
                    if self._logdir is not None:
                        steps, cur_loss, result, res, _ = sess.run([global_steps, loss, merged, res_sentence, train_op], feed_dict=feed)
                        writer.add_summary(result, steps)
                    else:
                        steps, cur_loss, res, _ = sess.run([global_steps, loss, res_sentence, train_op], feed_dict=feed)
                    yield sess, steps, cur_loss, x_batch, res

    @staticmethod
    def save(sess, path, steps=None):
        saver = tf.train.Saver()
        saver.save(sess, path, steps)

    @staticmethod
    def load(sess, checkpoint_dir):
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('./word_vecs.txt', binary=False)
    wv = model.vectors
    text_cnn = TextCNNAutoEncoder(100, wv, word_vec_dim=50, batch_size=10,
                                  filter_depth=20, logdir='./logdir', epochs=10, drop_prob=0.5, lr=0.01)

    for sess, steps, loss, x, res in text_cnn.fit('./index.txt'):
        if steps % 10 == 0:
            xs = x[:3].tolist()
            res = res[:3]
            for x, y in  zip(xs, res):
                print('input: %s' % ' '.join([model.index2word[i] for i in x]))
                print('output: %s' % ' '.join([model.index2word[i] for i in y]))

        if steps % 100 == 0:
            print('After %d(steps), loss = %f' % (steps, loss))
            TextCNNAutoEncoder.save(sess, './checkpoint/TextCNNEmbedding', steps=steps)
