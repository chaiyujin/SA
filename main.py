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

video_path = 'F:/dataset/GRID/video/s1/video/mpg_6000'
a_path = 'F:/dataset/GRID/video/s1/video/mpg_6000/normalized-bgbh5s.wav'
v_path = 'F:/dataset/GRID/video/s1/video/mpg_6000/bgbh5s.mpg'
video_path = 'test/video'
bs = 2
# process data
collector = ForNvidia(video_path)
train_data, valid_data = collector.collect(cache_path='test.pkl')
collector.pca_video_feature(0.99)
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
    if args.train:
        trainer.set_learning_rate(lr=1e-4, pca_lr=1e-12)
        trainer.init_variables(sess)
        trainer.train(sess, 200)
    else:
        trainer.restore(sess, 'save/my-model.pkl')

    for i in range(10):
        length = len(valid_data['path'])
        idx = np.random.randint(0, length, (1))[0]
        video_path = valid_data['path'][idx]['video_path']
        audio_slices, video_slices, a_path = collector.slice_media(video_path)
        video_name = os.path.splitext(video_path)[0].split('/')[-1]
        print(video_name)
        trainer.sample(sess, audio_slices, video_slices,
                       a_path, 'sample/' + video_name)
