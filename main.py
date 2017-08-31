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
video_path = 'test/per'
bs = 16
E = 8
# process data
collector = ForNvidia(video_path, video_ext='mp4')
train_data, valid_data, pca = collector.collect(scale=0.01, E=E, cache_path='per.pkl')
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
        net = Net(x, y, e, pca['components'], pca['mean'], drop=0.5)
        trainer = Handler(net, data_set, test_set)
        trainer.set_learning_rate(lr=1e-4, e_lr=1e-5, pca_lr=1e-14)
        trainer.init_variables(sess)
        trainer.train(sess, 300, 20)

        trainer.set_learning_rate(lr=1e-6, e_lr=1e-4, pca_lr=1e-12)
        trainer.init_variables(sess)
        trainer.load_e_vector()
        trainer.train(sess, 300, 20)
    else:
        net = Net(x, y, e, pca['components'], pca['mean'], drop=0)
        trainer = Handler(net, data_set, test_set)
        trainer.init_variables(sess)
        trainer.restore(sess, 'save_2/my-model.pkl')
        # trainer.load_train_set()
        trainer.load_e_vector()

    test_path = 'F:/dataset/RAVDESS/actor_01/speech/01-01-01-01-01-01-01.mp4'
    the_set = trainer.data_set_.data_
    import pickle
    with open('e_vector.pkl', 'rb') as file:
        the_set['e_vector'] = pickle.load(file)
    last_path = ''
    idx = 0
    for i in range(15):
        video_path = last_path
        while video_path == last_path:
            length = len(the_set['path'])
            idx += 1
            video_path = the_set['path'][idx]['video_path']
        right = idx + 1
        while the_set['path'][right]['video_path'] == video_path:
            right += 1
        e_vector = the_set['e_vector'][idx: right]
        print(e_vector)
        last_path = video_path
        video_path = the_set['path'][0]['video_path']
        # video_path = test_path
        audio_slices, video_slices, a_path = collector.slice_media(video_path)
        _, video_slices, a_path = collector.slice_media(the_set['path'][idx]['video_path'])
        video_name = os.path.splitext(video_path)[0].split('/')[-1]
        print(video_name)
        trainer.sample(sess, audio_slices, video_slices, e_vector,
                       a_path, 'sample/' + video_name + str(i))
