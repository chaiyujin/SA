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
video_path = 'test/emo_1'
bs = 4
E = 4
# process data
collector = ForNvidia(video_path, video_ext='mp4')
train_data, valid_data, pca = collector.collect(scale=0.1, E=E, cache_path='emo.pkl')
print(pca['mean'])
data_set = DataSet(train_data, train_data['len'], bs)
test_set = DataSet(valid_data, valid_data['len'], bs)
# print(data_set.random_batch())

# input tensor
if not args.train:
    bs = 1
x = tf.placeholder(tf.float32, [bs, 64, 32, 1], name='audio_lpc')
y = tf.placeholder(tf.float32, [bs, 36], name='mouth')
e = tf.placeholder(tf.float32, [bs, E], name='e_vector')


with tf.Session() as sess:
    if args.train:
        net = Net(x, y, e, pca['components'], pca['mean'], drop=0.3)
        trainer = Handler(net, data_set, test_set)
        trainer.set_learning_rate(lr=1e-4, e_lr=1e-8, pca_lr=1e-14)
        trainer.init_variables(sess)
        trainer.train(sess, 200, 30)

        # trainer.set_learning_rate(lr=1e-6, e_lr=1e-6, pca_lr=1e-12)
        # trainer.init_variables(sess)
        # trainer.train(sess, 500, 20)
    else:
        net = Net(x, y, e, pca['components'], pca['mean'], drop=0)
        trainer = Handler(net, data_set, test_set)
        trainer.init_variables(sess)
        trainer.restore(sess, 'save_1/my-model.pkl')
        # trainer.load_train_set()

    the_set = trainer.data_set_.data_
    import pickle
    with open('e_vector.pkl', 'rb') as file:
        the_set['e_vector'] = pickle.load(file)
    for i in range(2):
        length = len(the_set['path'])
        print(length)
        idx = np.random.randint(0, length, (1))[0]
        idx = i * 400
        e_vector = the_set['e_vector'][idx]
        print(e_vector)
        video_path = the_set['path'][idx]['video_path']
        audio_slices, video_slices, a_path = collector.slice_media(video_path)
        video_name = os.path.splitext(video_path)[0].split('/')[-1]
        print(video_name)
        trainer.sample(sess, audio_slices, video_slices, e_vector,
                       a_path, 'sample/' + video_name)
