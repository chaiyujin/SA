import sys
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


def loss_function(pred, true, e_vector, decay=0.99):
    def m(x, y):
        return tf.subtract(x, y)
    # 1. position term
    P = tf.reduce_mean(
        tf.square(tf.subtract(true, pred)),
        axis=0
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
        axis=0
    )
    M = tf.multiply(M, 2)
    # 3. evector
    evec_l = e_vector[:half_size]
    evec_r = e_vector[half_size:]
    R_ = tf.reduce_mean(
        tf.square(m(evec_l, evec_r)),
        axis=0
    )
    R_ = tf.multiply(R_, 2)
    nm = tf.reduce_mean(tf.square(e_vector))
    R = tf.divide(R_, nm)

    return P, M, R


class LossAdaptor():
    def __init__(self, decay=0.99):
        self.decay_ = decay
        self.beta_t_ = 1
        self.v_ = 0

    def adapt(self, loss):
        self.v_ =\
            self.decay_ * self.v_ +\
            (1 - self.decay_) * (loss ** 2).mean()
        self.beta_t_ *= self.decay_
        v_hat = self.v_ / (1 - self.beta_t_)
        return loss.mean() / (np.sqrt(v_hat) + 1e-8)


if __name__ == '__main__':
    import numpy as np
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
    