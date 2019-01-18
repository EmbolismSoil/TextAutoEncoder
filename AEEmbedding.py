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
        self._encoded = self._encoder()
        self._decode()

    def _encoder(self):
        x = tf.placeholder(tf.int64, shape=[None, self._sentence_len], name='embedding-layer-x')
        cur_batch_size = tf.shape(x)[0]
        x_emb = tf.placeholder(tf.float64, shape=[None, self._word_vec_dim], name='embedding-table')
        x_embedded = tf.nn.embedding_lookup(x_emb, x)
        x_embedded = tf.reshape(x_embedded, shape=[cur_batch_size, self._sentence_len, self._word_vec_dim, 1])
        setattr(self, 'x_embedded', x_embedded)

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

            features_len = self._sentence_len - filter_size + 1
            out, mask = tf.nn.max_pool_with_argmax(out, ksize=(1, features_len, 1, 1), padding='VALID', strides=[1, 1, 1, 1])
            masks.append(mask)
            conv_outs.append(out)

        concated = tf.concat(conv_outs, axis=3, name='concated')
        reshaped = tf.reshape(concated, shape=[cur_batch_size, 150], name='reshaped')
        encoded = reshaped

        #decoder
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
            pass

        return encoded

    def _decode(self):
        encoded = self._encoded
        deconv_outs = []
        for idx, filter_size in enumerate(self._filter_sizes):
            out = layers.conv2d_transpose(encoded,
                                          filters=self._filter_depth,
                                          kernel_size=(filter_size, self._sentence_len),
                                          activation=tf.nn.relu,
                                          padding='VALID',
                                          use_bias=True, name='deconv_layer_%d' % idx)
            # (batch_size, filter_depth, sentence_len, filter_depth)
            features_len = self._sentence_len + filter_size - 1
            out = layers.max_unpooling2d(out, pool_size=(features_len, 1), strides=(1, 1), padding='VALID', name='max_unpooling_%d' % idx)
            deconv_outs.append(out)
        pass


if __name__ == '__main__':
    text_cnn = TextCNNAutoEncoder(100, None)