from __future__ import absolute_import

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.layers as tflayers
from .adam import Adam
from tqdm import trange
from ..media.video.video_feature import draw_mouth_landmarks


def get_shape(tensor):
    return [int(d) for d in tensor.get_shape()]


def print_shape(tensor, output_name=None):
    if output_name is None:
        output_name = 'Tensor shape:'
    else:
        output_name += ' shape:'
    shape = get_shape(tensor)
    print(output_name, shape)


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def audio_abstraction_net(input):
    scope = sys._getframe().f_code.co_name
    with tf.variable_scope(scope):
        # input: 64 x 32 x 1
        layers_config = [
            {'num_outputs': 72, 'kernel_size': (1, 3),
             'stride': (1, 2), 'activation_fn': lrelu},
            {'num_outputs': 108, 'kernel_size': (1, 3),
             'stride': (1, 2), 'activation_fn': lrelu},
            {'num_outputs': 162, 'kernel_size': (1, 3),
             'stride': (1, 2), 'activation_fn': lrelu},
            {'num_outputs': 243, 'kernel_size': (1, 3),
             'stride': (1, 2), 'activation_fn': lrelu},
            {'num_outputs': 256, 'kernel_size': (1, 2),
             'stride': (1, 2), 'activation_fn': lrelu}
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
            {'num_outputs': 256, 'kernel_size': (3, 1),
             'stride': (2, 1), 'activation_fn': lrelu},
            {'num_outputs': 256, 'kernel_size': (3, 1),
             'stride': (2, 1), 'activation_fn': lrelu},
            {'num_outputs': 256, 'kernel_size': (3, 1),
             'stride': (2, 1), 'activation_fn': lrelu},
            {'num_outputs': 256, 'kernel_size': (3, 1),
             'stride': (2, 1), 'activation_fn': lrelu},
            {'num_outputs': 256, 'kernel_size': (4, 1),
             'stride': (4, 1), 'activation_fn': lrelu}
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


def output_net(anime_feature, init_pca, init_mean):
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
            'num_outputs': len(init_pca),
            'activation_fn': lrelu
        }
        pca_coeff = tflayers.fully_connected(**layer_config)
        print_shape(pca_coeff, '\tlayer0 output')
    # init the network with pca vectors
    with tf.variable_scope('pca_dense'):
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
    return pca_coeff, landmarks, var_list0, var_list1


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
    return tf.add(Lp, Lm)
    # return tf.add(tf.add(Lp, Lm), Lr)


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
    def __init__(self, input, output, e_vector, init_pca, init_mean):
        audio_feature, var_list0 = audio_abstraction_net(input)
        anime_feature, var_list1 = articulation_net(audio_feature, e_vector)
        pca_coeff, landmarks_pred, var_list2, var_list3 =\
            output_net(anime_feature, init_pca, init_mean)
        var_list0.extend(var_list1)
        var_list0.extend(var_list2)
        # input, output, e_vector
        self.input = input
        self.output = output
        self.e_vector = e_vector
        self.pred = landmarks_pred
        self.pca = pca_coeff
        # all var and pca coefficient
        self.var_list = var_list0
        self.pca_var = var_list3
        self.loss_fn_list = loss_function(landmarks_pred, output, e_vector)
        # regularized loss
        self.loss_regular_ = []
        for i, loss_fn in enumerate(self.loss_fn_list):
            self.loss_regular_.append(LossRegularizer())
        self.loss = regularize_loss(self.loss_fn_list, self.loss_regular_)
        # self.loss = tf.losses.mean_pairwise_squared_error(
        #     labels=self.output,
        #     predictions=self.pred
        # )
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(
            self.loss, var_list=self.var_list
        )
        self.pca_optimizer = tf.train.AdamOptimizer(1e-12).minimize(
            self.loss, var_list=self.pca_var
        )
        # gradient for e
        # self.grad_E = tf.gradients(self.loss, [e_vector])[0]
        # self.E_optimizer = Adam(1e-8)

        self.saver = tf.train.Saver()

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


def printProgressBar(iteration, total, prefix='',
                     suffix='', decimals=1, length=20, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


class Handler():
    def __init__(self, net, train_set, valid_set):
        self.data_set_ = train_set
        self.test_set_ = valid_set
        self.net_ = net

    def train(self, sess, n_epoches, mode='loop'):
        N_EARLY_STOP = 80
        self.sess = sess
        best_loss = 1e6
        early_step = N_EARLY_STOP
        epoch_list = []
        loss_lists = {
            'train_p': [],
            'valid_p': [],
            'train_m': [],
            'valid_m': []
        }
        for epoch in range(n_epoches):
            if mode == 'random':
                batch, indexes = self.data_set_.random_batch()
                train_result = self.__train_batch(batch, indexes,
                                                  self.data_set_)
                valid_result = self.__valid_batch(batch, indexes)
            elif mode == 'loop':
                train_result = self.__loop_set(self.data_set_, is_train=True)
                valid_result = self.__loop_set(self.test_set_, is_train=False)
            # 4. print the loss
            print('[' + str(epoch) + '/' + str(n_epoches) + ']', 'train loss:'
                  'P %.4f' % train_result[0].mean(),
                  'M %.4f' % train_result[1].mean(), '\tvalid loss:',
                  'P %.4f' % valid_result[0].mean(),
                  'M %.4f' % valid_result[1].mean())
            cur_loss = valid_result[0].mean()
            print(cur_loss, best_loss, early_step)
            if cur_loss < best_loss:
                best_loss = cur_loss
                self.net_.saver.save(sess, 'save/my-model.pkl')
                early_step = N_EARLY_STOP
            else:
                early_step -= 1
                if early_step == 0:
                    break
            # draw
            epoch_list.append(epoch)
            loss_lists['train_p'].append(train_result[0].mean())
            loss_lists['train_m'].append(train_result[1].mean())
            loss_lists['valid_p'].append(valid_result[0].mean())
            loss_lists['valid_m'].append(valid_result[1].mean())
            fig = plt.figure(figsize=(12, 12))
            p_plt = fig.add_subplot(211)
            m_plt = fig.add_subplot(212)
            p_plt.title.set_text('P loss')
            m_plt.title.set_text('M loss')
            p_plt.plot(epoch_list, loss_lists['train_p'], 'g', label='train')
            p_plt.plot(epoch_list, loss_lists['valid_p'], 'r', label='valid')
            m_plt.plot(epoch_list, loss_lists['train_m'], 'g', label='train')
            m_plt.plot(epoch_list, loss_lists['valid_m'], 'r', label='valid')
            p_plt.legend(prop={'size': 8})
            m_plt.legend(prop={'size': 8})
            plt.savefig('error.png')
            plt.clf()
            plt.close(fig)

            # self.__sample_batch(self.test_set_)

    def __sample_batch(self, data_set):
        data_set.reset_loop()
        batch, _ = data_set.next_batch()
        res = self.predict(self.sess, batch['input'], batch['e_vector'])
        bs = data_set.bs_
        for j in range(bs):
            pred = res[j]
            true = batch['output'][j]
            pred = np.reshape(pred, (18, 2))
            true = np.reshape(true, (18, 2))

            img = draw_mouth_landmarks(800, pred)
            img = draw_mouth_landmarks(800, true, (0, 0, 255), img, (0, 100))
            cv2.imshow('frame', img)
            cv2.waitKey(40)

    def __loop_set(self, data_set, is_train):
        data_set.reset_loop()
        total = data_set.length_loop()
        cnt = 0
        result = None
        for i in range(total):
            batch, indexes = data_set.next_batch()
            if is_train:
                tmp_res = self.__train_batch(batch, indexes, data_set)
            else:
                tmp_res = self.__valid_batch(batch, indexes)
            result = tmp_res if result is None else (tmp_res + result)
            cnt += 1
            loss_str = '%.4f\t%.4f' %\
                       (result[0].mean() / cnt,
                        result[1].mean() / cnt)
            printProgressBar(i, total - 1, loss_str)
        result /= cnt
        return result

    def __valid_batch(self, batch, indexes):
        feed_dict = self.net_.feed_dict(batch)
        to_run = []
        for i in range(len(self.net_.loss_fn_list)):
            to_run.append(self.net_.loss_fn_list[i])
        result = self.sess.run(to_run, feed_dict=feed_dict)
        return np.asarray([result[0], result[1]])

    def __train_batch(self, batch, indexes, data_set):
        # 1. calc loss function
        feed_dict = self.net_.feed_dict(batch)
        to_run = []
        for i in range(len(self.net_.loss_fn_list)):
            to_run.append(self.net_.loss_fn_list[i])
        to_run.extend([
            # self.net_.grad_E,
            self.net_.loss
        ])
        # to_run = [self.net_.pca, self.net_.loss]
        result = self.sess.run(to_run, feed_dict=feed_dict)
        # 2. optimize e vector
        # grad_E = result[-2]
        # # e vector optimizer
        # new_e = self.net_.E_optimizer.apply_gradient(
        #     batch['e_vector'], grad_E
        # )
        # data_set.adjust_e_vector(new_e, indexes)
        # net optimizer
        self.sess.run(
            [self.net_.optimizer],  #, self.net_.pca_optimizer],
            feed_dict=feed_dict
        )    
        # 3. update the loss regularizer
        self.net_.update_loss_regularizer(
            result[:len(self.net_.loss_fn_list)]
        )
        return np.asarray([result[0], result[1]])

    def predict(self, sess, input, e_vector):
        to_run = [self.net_.pred]
        feed_dict = {
            self.net_.input: input,
            self.net_.e_vector: e_vector
        }

        return sess.run(to_run, feed_dict=feed_dict)[0]

    def restore(self, sess, path='save/my-model.pkl'):
        self.net_.saver.restore(sess, path)


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
