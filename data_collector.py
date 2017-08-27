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
    def __init__(self, input_dir, output_dir,
                 video_ext='mpg', audio_wlen=0.016,
                 audio_wstep=0.008, audio_n_frame=64,
                 lpc_k=16, lpc_pre_e=None, feature_gap=2):
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._ext = video_ext
        self._k = lpc_k
        self._pre_e = lpc_pre_e
        self._wlen = audio_wlen
        self._wstep = audio_wstep
        self._aframe = audio_n_frame
        self._feature_gap = feature_gap
        self.pca = None

    def collect(self, loc=0.0, scale=0.01, E=24, keep_even=True, wait_key=-1):
        if os.path.exists(self._output_dir):
            with open(self._output_dir, 'rb') as file:
                res = pickle.load(file)
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

        with open(self._output_dir, 'wb') as f:
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
    collector = ForNvidia('Video/mpg', 'test')
    collector.collect(wait_key=-1)
    collector.pca_video_feature()
    # for data in collector.data_maps:
    #     for d in data['data']:
    #         lm = d['video']
    #         img = draw_mouth_landmarks(800, lm)
    #         x = collector.pca.transform(lm)
    #         lm = collector.pca.inverse_transform(x)
    #         img = draw_mouth_landmarks(800, lm, (0, 0, 255), img, (0, 100))
    #         cv2.imshow('mouth', img)
    #         cv2.waitKey(1)
    print(collector.data_list.shape)