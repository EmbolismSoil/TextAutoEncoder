import tensorflow as tf
from tensorflow import layers
import numpy as np


class TextCNNAutoEncoder(object):
    def __init__(self, sentence_len, wv_mat,
                 word_vec_dim=300, batch_size=100, filter_sizes=(2, 3, 4), filter_depth=50, drop_prob=0.5):
        self._sentence_len = sentence_len
        self._wv_mat = wv_mat
        self._word_vec_dim = word_vec_dim
        self._batch_size = batch_size
        self._filter_sizes = filter_sizes
        self._filter_depth = filter_depth
        self._drop_prob = drop_prob

        self._build_net()

    def _build_net(self):
        x = tf.placeholder(tf.int64, shape=[None, self._sentence_len], name='embedding-layer-x')
        dr = tf.placeholder(tf.float64, shape=[])
        cur_batch_size = tf.shape(x)[0]
        x_emb = tf.placeholder(tf.float64, shape=[None, self._word_vec_dim], name='embedding-table')
        x_embedded = tf.nn.embedding_lookup(x_emb, x)
        x_embedded = tf.reshape(x_embedded, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])
        setattr(self, 'x', x)
        setattr(self, 'x_emb', x_emb)
        setattr(self, 'dr', dr)

        # CNN Layers
        conv_outs = []
        masks = []
        for idx, filter_size in enumerate(self._filter_sizes):
            out = layers.conv2d(x_embedded,
                                filters=self._filter_depth,
                                kernel_size=(filter_size, self._word_vec_dim),
                                activation=tf.nn.relu, padding='VALID',
                                use_bias=True,
                                name='conv_layer_%d' % idx)

            out = layers.dropout(out, self._drop_prob)
            features_len = self._sentence_len - filter_size + 1
            out, mask = tf.nn.max_pool_with_argmax(out, ksize=(1, features_len, 1, 1), padding='VALID', strides=[1, 1, 1, 1])
            masks.append(mask)
            conv_outs.append(out)

        concated = tf.concat(conv_outs, axis=3, name='concated')
        reshaped = tf.reshape(concated, shape=[cur_batch_size, 150], name='reshaped')
        encoded = reshaped

        #decoder
        deconv_outs = []
        for i, (mask, out, filter_size) in enumerate(zip(masks, conv_outs, self._filter_sizes)):
            mask = tf.reshape(mask, shape=[cur_batch_size, self._filter_depth])
            idx = (mask - np.asarray(range(self._filter_depth))) / self._filter_depth
            features_len = self._sentence_len - filter_size + 1
            b = tf.range(tf.cast(cur_batch_size, dtype=tf.float64), delta=1.0, dtype=tf.float64)
            idx = idx - tf.transpose(b)*features_len
            idx = tf.cast(idx, tf.int64)
            idx = tf.one_hot(idx, depth=features_len, axis=1, dtype=tf.float64)
            idx = tf.reshape(idx, shape=[cur_batch_size, features_len, 1, self._filter_depth])
            unpool = tf.multiply(idx, out)

            deconv_out = layers.conv2d_transpose(unpool,
                                filters=self._filter_depth,
                                kernel_size=(filter_size, self._word_vec_dim),
                                activation=tf.nn.relu, padding='VALID',
                                use_bias=True,
                                name='deconv_layer_%d' % i)
            deconv_out = layers.dropout(deconv_out, self._drop_prob)
            deconv_outs.append(deconv_out)

        res_concated = tf.concat(deconv_outs, axis=3)
        res_sentence = tf.reduce_mean(res_concated, axis=3)
        res_sentence = tf.reshape(res_sentence, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])
        loss = tf.losses.cosine_distance(x_embedded, res_sentence, axis=2)

        setattr(self, 'encoded', encoded)
        setattr(self, 'loss', loss)

    def make_nonshuffle_dataset(self, data_path, batch_size, sep='\t'):
        dataset = tf.data.TextLineDataset(data_path)
        def _parse_line(line):
            items = tf.string_split([line], delimiter=sep).values
            c, ws = items[0], items[1]
            ws = tf.string_split([ws], delimiter=' ').values
            ws = tf.string_to_number(ws, tf.int32)
            ws = tf.cond(tf.size(ws) > self._sentence_len, lambda: tf.slice(ws, [0], [self._sentence_len]), lambda: ws)
            c = tf.string_to_number(c, tf.int32)
            return c, ws

        dataset = dataset.map(_parse_line)
        padded_shapes = (tf.TensorShape([]), tf.TensorShape([self._sentence_len]))
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

    def make_dataset(self, data_path, sep='\t'):
        dataset = self.make_nonshuffle_dataset(data_path, self._batch_size, sep)
        return dataset.shuffle(10000)


if __name__ == '__main__':
    text_cnn = TextCNNAutoEncoder(100, None)