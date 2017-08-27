from __future__ import absolute_import

import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from .adam import Adam


def get_shape(tensor):
    return [int(d) for d in tensor.get_shape()]


def print_shape(tensor, output_name=None):
    if output_name is None:
        output_name = 'Tensor shape:'
    else:
        output_name += ' shape:'
    shape = get_shape(tensor)
    print(output_name, shape)


def audio_abstraction_net(input):
    scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope):
        # input: 64 x 32 x 1
        layers_config = [
            {'num_outputs': 72, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 108, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 162, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 243, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 256, 'kernel_size': (1, 2), 'stride': (1, 2)}
        ]

        print(scope + ' {')
        print_shape(input, '\tinput')
        print()
        layer = [input]
        for i, layer_config in enumerate(layers_config):
            output = tflayers.conv2d(layer[i], **layer_config)
            layer.append(output)
            # print shape
            print_shape(output, '\tlayer' + str(i) + ' output')
        print('}')

    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=scope
    )
    return layer[-1], var_list


def articulation_net(audio_feature, e_vector):
    def concat_kernel(x, y):
        return x
        tile = get_shape(x)
        tile[0] = 1
        tile[-1] = 1
        y_expand = tf.expand_dims(y, 1)
        y_expand = tf.expand_dims(y_expand, 1)
        y_tiled = tf.tile(y_expand, tile)
        return tf.concat((x, y_tiled), axis=3)

    scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope):
        # input: 32 x 1 x (256 + E)
        layers_config = [
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (4, 1), 'stride': (4, 1)}
        ]

        print(scope + ' {')
        print_shape(audio_feature, '\tinput')
        print_shape(e_vector, '\te_vector')
        print()
        layer = [audio_feature]
        for i, layer_config in enumerate(layers_config):
            # conv2d
            output = tflayers.conv2d(layer[i], **layer_config)
            # concat with e_vector
            concated = concat_kernel(output, e_vector)
            layer.append(concated)
            # print shape
            print_shape(concated, '\tlayer' + str(i) + ' output')
        print('}')

    var_list = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=scope
    )
    return layer[-1], var_list


def output_net(anime_feature, init_pca_vectors):
    print('output_net {')
    old_shape = get_shape(anime_feature)
    new_shape = [old_shape[0], old_shape[-1]]
    anime_feature = tf.reshape(
        anime_feature, new_shape
    )
    print_shape(anime_feature, '\tinput')
    print()
    # 1. anime_feature -> pca coefficients
    with tf.variable_scope('anime_dense'):
        layer_config = {
            'inputs': anime_feature,
            'num_outputs': len(init_pca_vectors),
            'activation_fn': None
        }
        pca_coeff = tflayers.fully_connected(**layer_config)
        print_shape(pca_coeff, '\tlayer0 output')
    # init the network with pca vectors
    with tf.variable_scope('pca_dense'):
        init = tf.constant_initializer(value=init_pca_vectors)
        layer_config = {
            'inputs': pca_coeff,
            'num_outputs': init_pca_vectors.shape[1],
            'activation_fn': None,
            'weights_initializer': init
        }
        landmarks = tflayers.fully_connected(**layer_config)
        print_shape(landmarks, '\tfinal output')
    print('}')

    var_list0 = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='anime_dense'
    )
    var_list1 = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope='pca_dense'
    )
    return landmarks, var_list0, var_list1


def loss_function(pred, true, e_vector):
    def m(x, y):
        return tf.subtract(x, y)
    # 1. position term
    P = tf.reduce_mean(
        tf.square(tf.subtract(true, pred)),
        axis=1
    )
    # 2. movement term
    shape = get_shape(pred)
    half_size = int(shape[0] / 2)
    pred_l = pred[:half_size]
    pred_r = pred[half_size:]
    true_l = true[:half_size]
    true_r = true[half_size:]
    M = tf.reduce_mean(
        tf.square(
            tf.subtract(
                m(pred_l, pred_r),
                m(true_l, true_r)
            )
        ),
        axis=1
    )
    M = tf.multiply(M, 2)
    # 3. evector
    evec_l = e_vector[:half_size]
    evec_r = e_vector[half_size:]
    R_ = tf.reduce_mean(
        tf.square(m(evec_l, evec_r)),
        axis=1
    )
    R_ = tf.multiply(R_, 2)
    nm = tf.reduce_mean(tf.square(e_vector))
    R = tf.divide(R_, nm)

    return P, M, R


def regularize_loss(loss_list, reg_list):
    Lp = tf.multiply(
        tf.reduce_mean(loss_list[0]),
        reg_list[0].scale
    )
    Lm = tf.multiply(
        tf.reduce_mean(loss_list[1]),
        reg_list[1].scale
    )
    Lr = tf.multiply(
        tf.reduce_mean(loss_list[2]),
        reg_list[2].scale
    )
    return tf.add(tf.add(Lp, Lm), Lr)


class LossRegularizer():
    def __init__(self, decay=0.99):
        self.decay_ = decay
        self.beta_t_ = 1
        self.v_ = 0
        self.scale = tf.placeholder(tf.float32, [1])
        self.feed_dict = {
            self.scale: np.asarray([1], dtype=np.float32)
        }

    def update(self, loss):
        self.v_ =\
            self.decay_ * self.v_ +\
            (1 - self.decay_) * (loss ** 2).mean()
        self.beta_t_ *= self.decay_
        v_hat = self.v_ / (1 - self.beta_t_)
        # update feed_dict
        self.feed_dict = {
            self.scale: np.asarray(
                [1 / (np.sqrt(v_hat) + 1e-8)],
                dtype=np.float32
            )
        }


class Net():
    def __init__(self, input, output, e_vector, init_pca_vectors):
        audio_feature, var_list0 = audio_abstraction_net(input)
        anime_feature, var_list1 = articulation_net(audio_feature, e_vector)
        landmarks_pred, var_list2, var_list3 =\
            output_net(anime_feature, init_pca_vectors)
        var_list0.extend(var_list1)
        var_list0.extend(var_list2)
        # input, output, e_vector
        self.input = input
        self.output = output
        self.e_vector = e_vector
        self.pred = landmarks_pred
        # all var and pca coefficient
        self.var_list = var_list0
        self.pca_coeff = var_list3
        self.loss_fn_list = loss_function(landmarks_pred, output, e_vector)
        # regularized loss
        self.loss_regular_ = []
        for i, loss_fn in enumerate(self.loss_fn_list):
            self.loss_regular_.append(LossRegularizer())
        self.loss = regularize_loss(self.loss_fn_list, self.loss_regular_)

        self.loss = tf.losses.absolute_difference(
            labels=self.output,
            predictions=self.pred
        )
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(
            self.loss, var_list=self.var_list
        )
        self.pca_optimizer = tf.train.AdamOptimizer(1e-4).minimize(
            self.loss, var_list=self.pca_coeff
        )
        # gradient for e
        self.grad_E = tf.gradients(self.loss, [e_vector])[0]
        self.E_optimizer = Adam(1e-8)

    def feed_dict(self, batch):
        feed = {
            self.input: batch['input'],
            self.output: batch['output'],
            self.e_vector: batch['e_vector']
        }
        for reg in self.loss_regular_:
            for k in reg.feed_dict:
                if k in feed:
                    raise Exception('Impossible feed_dict')
                feed[k] = reg.feed_dict[k]
        return feed

    def update_loss_regularizer(self, loss_list):
        for reg, loss in zip(self.loss_regular_, loss_list):
            reg.update(loss)


class Trainer():
    def __init__(self, net, data_set):
        self.data_set_ = data_set
        self.net_ = net

    def train(self, sess, n_epoches):
        for epoch in range(n_epoches):
            batch, indexes = self.data_set_.random_batch()
            # 1. calc loss function
            feed_dict = self.net_.feed_dict(batch)
            to_run = []
            for i in range(len(self.net_.loss_fn_list)):
                to_run.append(self.net_.loss_fn_list[i])
            to_run.extend([
                self.net_.grad_E,
                self.net_.loss
            ])
            to_run = [self.net_.loss]
            result = sess.run(to_run, feed_dict=feed_dict)
            # 2. optimize e vector
            # grad_E = result[-2]
            # # e vector optimizer
            # new_e = self.net_.E_optimizer.apply_gradient(
            #     batch['e_vector'], grad_E
            # )
            # self.data_set_.adjust_e_vector(new_e, indexes)
            # net optimizer
            sess.run(
                [self.net_.optimizer, self.net_.pca_optimizer],
                feed_dict=feed_dict
            )    
            # 3. update the loss regularizer
            self.net_.update_loss_regularizer(
                result[:len(self.net_.loss_fn_list)]
            )
            # 4. print the loss
            epoch_loss = result[-1]
            print('[' + str(epoch) + '/' + str(n_epoches) + ']', epoch_loss)

    def predict(self, sess, input, e_vector):
        to_run = [self.net_.pred]
        feed_dict = {
            self.net_.input: input,
            self.net_.e_vector: e_vector
        }

        return sess.run(to_run, feed_dict=feed_dict)[0]


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [1, 64, 32, 1])
    e = tf.placeholder(tf.float32, [24])
    init_pca_vectors = np.random.random((150, 18 * 2))
    init_pca_vectors = np.asarray(init_pca_vectors, np.float32)

    audio_feature, var_list0 = audio_abstraction_net(x)
    anime_feature, var_list1 = articulation_net(audio_feature, e)
    landmarks_pred, var_list2, var_list3 =\
        output_net(anime_feature, init_pca_vectors)

    var_list0.extend(var_list1)
    var_list0.extend(var_list2)
