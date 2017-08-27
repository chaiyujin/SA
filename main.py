import cv2
import numpy as np
import tensorflow as tf
from data_collector import ForNvidia
from model.data_set import DataSet
from model.nv_net import Net, Trainer
from Video.video_feature import draw_mouth_landmarks


bs = 8
# process data
collector = ForNvidia('Video/mpg', 'test')
all_data, data_size = collector.collect()
collector.pca_video_feature()
print(collector.pca.components_.shape)
print(all_data['input'][0].shape)
print(all_data['output'][0].shape)
data_set = DataSet(all_data, data_size, bs)

# input tensor
x = tf.placeholder(tf.float32, [bs, 64, 32, 1])
y = tf.placeholder(tf.float32, [bs, 36])
e = tf.placeholder(tf.float32, [bs, 24])

net = Net(x, y, e, collector.pca.components_)
trainer = Trainer(net, data_set)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    trainer.train(sess, 10000)

    for i in range(int(data_size / bs)):
        res = trainer.predict(
            sess,
            all_data['input'][i * bs: (i + 1) * bs],
            all_data['e_vector'][i * bs: (i + 1) * bs]
        )
        print(res.shape)
        for j in range(bs):
            pred = res[j]
            true = all_data['output'][i * bs + j]
            pred = np.reshape(pred, (18, 2))
            true = np.reshape(true, (18, 2))

            img = draw_mouth_landmarks(800, pred)
            img = draw_mouth_landmarks(800, true, (0, 0, 255), img, (0, 100))
            cv2.imshow('frame', img)
            cv2.waitKey()