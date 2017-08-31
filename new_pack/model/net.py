from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tflayers


def get_shape(tensor):
    return [int(d) for d in tensor.get_shape()]


def print_shape(tensor, output_name=None):
    if output_name is None:
        output_name = 'Tensor shape:'
    else:
        output_name += ' shape:'
    shape = get_shape(tensor)
    print(output_name, shape)


def network(input, e_vector, init_pca, init_mean,
            is_train, audio_drop, anime_drop):
    # Audio_Abstract
    scope = 'Audio_Abstract'
    var_list = []
    pca_list = []
    with tf.variable_scope(scope):
        layers_config = [
            {'num_outputs': 72, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 108, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 162, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 243, 'kernel_size': (1, 3), 'stride': (1, 2)},
            {'num_outputs': 256, 'kernel_size': (1, 2), 'stride': (1, 2)}
        ]
        print_shape(input, 'input')
        layer = [input]
        for i, l_config in enumerate(layers_config):
            output = tflayers.conv2d(layer[i], **l_config)
            output = tf.cond(
                is_train,
                lambda: tflayers.dropout(output, keep_prob=1. - audio_drop),
                lambda: tf.identity(output)
            )
            layer.append(output)
            print_shape(output, 'audio_layer' + str(i) + ' output')
        audio_feature = layer[-1]
        # var list
        var_list.extend(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope
        ))

    scope = 'Anime_Generator'
    with tf.variable_scope(scope):
        layers_config = [
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (3, 1), 'stride': (2, 1)},
            {'num_outputs': 256, 'kernel_size': (4, 1), 'stride': (4, 1)}
        ]
        print()
        layer = [audio_feature]
        for i, l_config in enumerate(layers_config):
            output = tflayers.conv2d(layer[i], **l_config)
            output = tf.cond(
                is_train,
                lambda: tflayers.dropout(output, keep_prob=1. - audio_drop),
                lambda: tf.identity(output)
            )
            layer.append(output)
            print_shape(output, 'anime_layer' + str(i) + 'output')
        anime_feature = layer[-1]
        # var list
        var_list.extend(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope
        ))

    print()
    scope = 'Anime_2_PCA'
    with tf.variable_scope(scope):
        layer_config = {
            'inputs': anime_feature,
            'num_outputs': len(init_pca),
            'activation_fn': None
        }
        pca_coeff = tflayers.fully_connected(**layer_config)
        print_shape(pca_coeff, 'dense_layer0 output')
        # var list
        var_list.extend(tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope
        ))

    scope = 'PCA_2_LM'
    with tf.variable_scope(scope):
        init_w = tf.constant_initializer(value=init_pca)
        init_b = tf.constant_initializer(value=init_mean)
        layer_config = {
            'inputs': pca_coeff,
            'num_outputs': init_pca.shape[1],
            'activation_fn': None,
            'weights_initializer': init_w,
            'biases_initializer': init_b
        }
        landmarks = tflayers.fully_connected(**layer_config)
        print_shape(landmarks, 'final output')
        # var list
        pca_list = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope
        )

    return landmarks, var_list, pca_list


def loss_function(pred, true, e_vector):
    def m(x, y):
        return tf.subtract(x, y)

    # 1. position term
    with tf.variable_scope('P'):
        P = tf.reduce_mean(
            tf.square(tf.subtract(true, pred)),
            axis=1,
            name='loss_p'
        )
    with tf.variable_scope('M'):
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
        M = tf.multiply(M, 2, name='loss_m')
    with tf.variable_scope('R'):
        # 3. evector
        evec_l = e_vector[:half_size]
        evec_r = e_vector[half_size:]
        R_ = tf.reduce_mean(
            tf.square(m(evec_l, evec_r)),
            axis=1
        )
        R_ = tf.multiply(R_, 2)
        nm = tf.reduce_mean(tf.square(e_vector))
        R = tf.divide(R_, nm, name='loss_r')

    return P, M, R


class LossRegularizer():
    def __init__(self, name, decay=0.99):
        self.decay_ = decay
        self.beta_t_ = 1
        self.v_ = 0
        self.scale = tf.placeholder(tf.float32, [1], name='scale' + name)
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


def regularize_loss(loss_list, reg_list):
    Lp = tf.multiply(
        tf.reduce_mean(loss_list[0]),
        reg_list[loss_list[0].name].scale
    )
    Lm = tf.multiply(
        tf.reduce_mean(loss_list[1]),
        reg_list[loss_list[1].name].scale
    )
    Lr = tf.multiply(
        tf.reduce_mean(loss_list[2]),
        reg_list[loss_list[2].name].scale
    )
    return tf.add(tf.add(Lp, Lm), Lr, name='RegLoss')


class Net():
    def __init__(self, config):
        bs = config['batch_size'] if config['train'] else 1
        self.x = tf.placeholder(tf.float32, [bs, 64, 32, 1])
        self.y = tf.placeholder(tf.float32, [bs, 36])
        self.e = tf.placeholder(tf.float32, [bs, config['E']])
        self.t = tf.placeholder(tf.bool, [])
        # network
        landmarks, var_list, pca_list = network(
            self.x, self.e, config['init_pca'], config['init_mean'],
            self.t, config['audio_drop'], config['anime_drop']
        )
        self.var_list = var_list
        self.pca_list = pca_list
        self.pred = landmarks
        self.true = self.y
        # loss
        self.loss_fns = loss_function(self.pred, self.true, self.e)
        # regulazrized loss
        self.loss_reg = {}
        for loss_fn in self.loss_fns:
            self.loss_reg[loss_fn.name] = LossRegularizer(loss_fn.name)
        # real loss
        self.loss = regularize_loss(self.loss_fns, self.loss_reg)
        # gradient for e
        self.grad_e = tf.gradients(self.loss, self.e)[0]
        # learning rate for Adam
        self.lr = tf.placeholder(tf.float32, [])
        self.lr_pca = tf.placeholder(tf.float32, [])

        # training phase and other things
        self.train_id = 0
        # save for net
        self.saver = tf.train.Saver()

    def train(self, sess, config, ignore=False):
        self.sess = sess
        # new training phase
        self.train_id += 1
        # load trained model
        need_restore = False
        if not ignore:
            if os.path.exists(self.save_path['model']):
                print('Training phase', self.train_phase, 'is done.')
                return
            else:
                need_restore = True
        # init optimizer
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(
            self.loss, var_list=self.var_list
        )
        self.pca_optim = tf.train.AdamOptimizer(self.lr_pca).minimize(
            self.loss, var_list=self.pca_list
        )
        self.e_optim = Adam(config['lr_e'])
        # run init
        sess.run(tf.global_variables_initializer())
        if need_restore:
            print('Restore from previous phase.')
            self.restore(self.restore_path)
        self.__train(config)

    def __train(self, config):
        print('Begin to training phase', self.train_phase)
        

    def save(self, path):
        self.saver.save(self.sess, path['model'])

    def restore(self, path):
        self.saver.restore(self.sess, path['model'])

    @property
    def train_phase(self):
        return str(self.train_id)

    @property
    def save_path(self):
        path = 'save_' + self.train_phase + '/'
        if not os.path.exists(path):
            os.mkdir(path)
        return {
            'model': path + 'model',
            'e_vec': path + 'e_vec.pkl'
        }

    @property
    def restore_path(self):
        prefix = 'save_' + str(self.train_id - 1) + '/'
        return {
            'model': prefix + 'model',
            'e_vec': prefix + 'e_vec.pkl'
        }
