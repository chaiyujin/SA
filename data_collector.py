import os
import cv2
import pickle
import numpy as np
from sklearn.decomposition import PCA
from utils.dir import find_files
from video_process import pre_process_video
from video_process import generate_video_from_landmarks
from video_process import mux
from Video.video_feature import draw_mouth_landmarks


class ForNvidia():
    def __init__(self, input_dir,
                 video_ext='mpg', audio_wlen=0.016,
                 audio_wstep=0.008, audio_n_frame=64,
                 lpc_k=16, lpc_pre_e=None, feature_gap=2):
        self._input_dir = input_dir
        self._ext = video_ext
        self._k = lpc_k
        self._pre_e = lpc_pre_e
        self._wlen = audio_wlen
        self._wstep = audio_wstep
        self._aframe = audio_n_frame
        self._feature_gap = feature_gap
        self.pca = None

    def collect(self, loc=0.0, scale=0.01, E=24,
                keep_even=True, wait_key=-1,
                cache_path=None, force_replace=False):
        if (not force_replace) and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                res = pickle.load(file)
                self.data_map = res[0]
                self._all_video_feature = res[2]
                return res[0], res[1]
        print('Collecting data from:', self._input_dir)
        self.data_map = {
            'input': [],
            'output': [],
            'e_vector': [],
            'path': []
        }
        self._all_video_feature = []
        lists = find_files(self._input_dir, self._ext)
        for count, path in enumerate(lists):
            print('[' + str(count) + '/' + str(len(lists)) + ']')
            result = pre_process_video(
                        path, 'lpc', winlen=self._wlen,
                        winstep=self._wstep, k=self._k, pre_e=self._pre_e)
            # 520ms => 64 * 8ms
            data = []
            length = len(result['audio_feature']) - self._aframe
            if keep_even:
                if int(length / self._feature_gap) & 1 == 0:
                    length -= self._feature_gap
            for i in range(0, length, self._feature_gap):
                l = i
                r = l + self._aframe
                m = (l + r) >> 1
                if (l + r) & 1 == 1:
                    mid =\
                        (result['video_feature'][m] +
                         result['video_feature'][m + 1]) / 2
                else:
                    mid = result['video_feature'][m]
                # move to center
                mid -= mid.mean()
                # save in the all video feature
                self._all_video_feature.append(mid.flatten())
                # save in the data
                data.append({
                    'video': np.asarray(mid, np.float32),
                    'audio': np.asarray(
                        result['audio_feature'], np.float32)[l: r, 1:]
                })
                # show the mouth
                if wait_key >= 0:
                    img = draw_mouth_landmarks(800, mid)
                    cv2.imshow('mouth', img)
                    cv2.waitKey(wait_key)
            print('->Get', len(data))
            e_vector = np.random.normal(loc=loc, scale=scale, size=(E))
            for i in range(len(data)):
                self.data_map['input'].append(
                    np.expand_dims(data[i]['audio'], -1))
                self.data_map['output'].append(data[i]['video'].flatten())
                self.data_map['e_vector'].append(e_vector)
                self.data_map['path'].append(result['path'])

        for k in self.data_map:
            self.data_map[k] = np.asarray(self.data_map[k])

        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(
                    (
                        self.data_map,
                        len(self.data_map['input']),
                        self._all_video_feature
                    ),
                    f
                )
        return self.data_map, len(self.data_map['input'])

    def pca_video_feature(self):
        self.pca = PCA(n_components=0.97, svd_solver='full')
        self.pca.fit(self._all_video_feature)


if __name__ == '__main__':
    import tensorflow as tf
    import tensorflow.contrib.layers as tflayers

    def pca_net(pca_coeff, init_pca, init_mean):
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
            # print_shape(landmarks, '\tfinal output')
        return landmarks

    pca_coeff = tf.placeholder(tf.float32, (1, 3))

    collector = ForNvidia('test/video')
    collector.collect(wait_key=-1, cache_path='test.pkl')
    collector.pca_video_feature()
    print(collector.pca.components_.shape)

    landmarks = pca_net(pca_coeff, collector.pca.components_, collector.pca.mean_)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(len(collector.data_map['input'])):
            if True:
                d = collector.data_map
                lm = d['output'][i]
                img = draw_mouth_landmarks(800, np.reshape(lm, (18, 2)))
                x = collector.pca.transform(np.reshape(lm, (1, 36)))
                print(x)

                lm = sess.run([landmarks], {pca_coeff: x})[0]
                print(lm.shape)
                # lm = collector.pca.inverse_transform(x)
                img = draw_mouth_landmarks(800, np.reshape(lm, (18, 2)), (0, 0, 255), img, (0, 100))
                cv2.imshow('mouth', img)
                cv2.waitKey(1)
    # print(collector.data_list.shape)