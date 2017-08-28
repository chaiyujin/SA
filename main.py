import cv2
import numpy as np
import tensorflow as tf
from pack.data.data_collector import ForNvidia
from pack.data.data_set import DataSet
from pack.model.nv_net import Net, Handler
from pack.media.video.video_feature import draw_mouth_landmarks


video_path = 'F:/dataset/GRID/video/s1/video/mpg_6000'
# video_path = 'test/video'
bs = 32
# process data
collector = ForNvidia(video_path)
train_data, valid_data = collector.collect(cache_path='data_s1.pkl')
collector.pca_video_feature()
print(collector.pca.components_.shape)
print(train_data['input'][0].shape)
print(train_data['output'][0].shape)
data_set = DataSet(train_data, train_data['len'], bs)
test_set = DataSet(valid_data, valid_data['len'], bs)
# print(data_set.random_batch())

# input tensor
x = tf.placeholder(tf.float32, [bs, 64, 32, 1])
y = tf.placeholder(tf.float32, [bs, 36])
e = tf.placeholder(tf.float32, [bs, 24])

net = Net(x, y, e, collector.pca.components_, collector.pca.mean_)
trainer = Handler(net, data_set, test_set)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    trainer.train(sess, 800)

    # trainer.restore(sess, 'save/my-model.pkl')
    for i in range(int(train_data['len'] / bs)):
        res = trainer.predict(
            sess,
            train_data['input'][i * bs: (i + 1) * bs],
            train_data['e_vector'][i * bs: (i + 1) * bs]
        )
        print(res.shape)
        for j in range(bs):
            pred = res[j]
            true = train_data['output'][i * bs + j]
            pred = np.reshape(pred, (18, 2))
            true = np.reshape(true, (18, 2))

            img = draw_mouth_landmarks(800, pred)
            img = draw_mouth_landmarks(800, true, (0, 0, 255), img, (0, 100))
            cv2.imshow('frame', img)
            cv2.waitKey()
