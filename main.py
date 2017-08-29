import os
import argparse
import numpy as np
import tensorflow as tf
from pack.data.data_collector import ForNvidia
from pack.data.data_set import DataSet
from pack.model.nv_net import Net, Handler

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', action='store_true')
args = parser.parse_args()

video_path = 'F:/dataset/GRID/video/s12/video/mpg_6000'
# video_path = 'test/video'
bs = 32
# process data
collector = ForNvidia(video_path)
train_data, valid_data = collector.collect(cache_path='data_12.pkl')
collector.pca_video_feature(0.99)
data_set = DataSet(train_data, train_data['len'], bs)
test_set = DataSet(valid_data, valid_data['len'], bs)
# print(data_set.random_batch())

# input tensor
x = tf.placeholder(tf.float32, [bs, 64, 32, 1])
y = tf.placeholder(tf.float32, [bs, 36])
e = tf.placeholder(tf.float32, [bs, 24])


with tf.Session() as sess:
    if args.train:
        net = Net(x, y, e, collector.pca.components_,
                  collector.pca.mean_, 0.5)
        trainer = Handler(net, data_set, test_set)
        trainer.set_learning_rate(lr=1e-4, pca_lr=1e-12)
        trainer.init_variables(sess)
        trainer.train(sess, 800)
    else:
        net = Net(x, y, e, collector.pca.components_,
                  collector.pca.mean_, 0)
        trainer = Handler(net, data_set, test_set)
        trainer.restore(sess, 'save/my-model.pkl')

    for i in range(10):
        length = len(train_data['path'])
        idx = np.random.randint(0, length, (1))[0]
        video_path = train_data['path'][idx]['video_path']
        audio_slices, video_slices, a_path = collector.slice_media(video_path)
        video_name = os.path.splitext(video_path)[0].split('/')[-1]
        print(video_name)
        trainer.sample(sess, audio_slices, video_slices,
                       a_path, 'sample/' + video_name)
