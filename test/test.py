import tensorflow as tf

x = tf.constant([1], tf.float32)
W = tf.Variable([0], True, dtype=tf.float32)
b = tf.Variable([0], True, dtype=tf.float32)
y_ = tf.add(tf.multiply(x, W), b)
y = tf.constant([100], tf.float32)
loss_fn = tf.reduce_mean(tf.square(tf.subtract(y, y_)))
optim = tf.train.AdamOptimizer()

gradients = optim.compute_gradients(
    loss=loss_fn, var_list=[W, b])



for grad in gradients:
    print(grad[0])
    print(grad[1])
    print()