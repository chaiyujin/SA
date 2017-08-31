import os
import cv2
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import PCA
from ..utils.dir import find_files
from ..media.media_process import pre_process_video, upsample_video_feature
from ..media.media_process import generate_video_from_landmarks
from ..media.media_process import mux, demux_video
from ..media.video.video_feature import draw_mouth_landmarks
from ..media.audio.feature.audio_feature import lpc_feature


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

    def slice_media(self, video_path, fps=25.0):
        path = demux_video(video_path)
        audio_path = path['audio_path']
        video_path = path['video_path']
        # video
        lm_feature = upsample_video_feature(video_path, 1)
        lm_feature -= lm_feature.mean()
        # audio
        rate, audio = wavfile.read(audio_path)
        pad_length = int(rate * 0.26)
        zeros = np.zeros((pad_length))
        audio = np.concatenate((zeros, audio), axis=0)
        audio = np.concatenate((audio, zeros), axis=0)
        a_feature = lpc_feature(audio, rate,
                                winlen=self._wlen, winstep=self._wstep,
                                k=self._k, preemphsis=self._pre_e)
        feature_gap = int(1 / (fps * self._wstep))
        length = len(a_feature) - self._aframe
        a_feature = np.asarray(a_feature, np.float32)
        slices = []
        for i in range(0, length, feature_gap):
            l = i
            r = l + self._aframe
            slices.append(a_feature[l: r, 1:])
        slices = np.expand_dims(np.asarray(slices), -1)
        return slices, lm_feature, audio_path

    def collect(self, pca_percentage=0.99,
                loc=0.0, scale=0.01, E=24,
                keep_even=True, wait_key=-1,
                cache_path=None, force_replace=False):
        if (not force_replace) and cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as file:
                res = pickle.load(file)
                self.train_data = res[0]
                self.valid_data = res[1]
                self.pca_result = res[2]
                return res[0], res[1], res[2]
        print('Collecting data from:', self._input_dir)
        self.train_data = {
            'input': [], 'output': [],
            'e_vector': [], 'path': [], 'len': 0
        }
        self.valid_data = {
            'input': [], 'output': [],
            'e_vector': [], 'path': [], 'len': 0
        }
        self._all_video_feature = []
        lists = find_files(self._input_dir, self._ext)
        for count, path in enumerate(lists):
            print('[' + str(count) + '/' + str(len(lists)) + ']')

            if np.random.rand(1)[0] < 0.2:
                data_map = self.valid_data
            else:
                data_map = self.train_data

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
            e_prefix = np.random.uniform(-1.0, 1.0, size=(E - 2))
            for i in range(int(len(data) / 2)):
                for j in range(2):
                    e_vector = np.random.normal(loc=loc, scale=0.1, size=(2))
                    e_vector = np.concatenate((e_prefix, e_vector))
                    # print(e_vector.shape)
                    data_map['input'].append(
                        np.expand_dims(data[i * 2 + j]['audio'], -1))
                    data_map['output'].append(
                        data[i * 2 + j]['video'].flatten())
                    data_map['e_vector'].append(e_vector)
                    data_map['path'].append(result['path'])
                    data_map['len'] += 1

        for k in self.train_data:
            if k == 'path' or k == 'len':
                continue
            self.train_data[k] = np.asarray(self.train_data[k])
            self.valid_data[k] = np.asarray(self.valid_data[k])

        self.pca = PCA(n_components=pca_percentage, svd_solver='full')
        self.pca.fit(self._all_video_feature)
        self.pca_result = {
            'components': self.pca.components_,
            'mean': self.pca.mean_
        }
        print('Train data:', self.train_data['len'])
        print('Valid data:', self.valid_data['len'])

        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(
                    (
                        self.train_data,
                        self.valid_data,
                        self.pca_result
                    ),
                    f
                )
        return self.train_data, self.valid_data, self.pca_result


if __name__ == '__main__':
    collector = ForNvidia('test/video')
    collector.collect()

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

        for i in range(len(collector.train_data['input'])):
            if True:
                d = collector.train_data
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