from __future__ import division
from __future__ import absolute_import

import os
import cv2
import ffmpy
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from .audio.feature.audio_feature import lpc_feature, rms_normalize
from .audio.feature.audio_feature import formant
from .video.video_feature import init_dde, landmark_feature
from .video.video_feature import draw_mouth_landmarks

W = 800
VIDEO_FPS = 25.0
LOG_LEVEL = 'error'


# for generating data
def remove_files(lists):
    for path in lists:
        if os.path.exists(path):
            os.remove(path)


def demux_video(video_path, clear_old=False):
    global VIDEO_FPS

    if not os.path.exists(video_path):
        return None
    file_prefix, _ = os.path.splitext(video_path)
    print('Demuxing', video_path)
    v_path = file_prefix + '_video.mp4'
    w_path = file_prefix + '_audio.wav'
    if clear_old:
        remove_files([v_path, w_path])

    # 1. demux audio
    # rms normalize the audio
    true, w_path = rms_normalize(video_path)
    if not true:
        return None
    # 2. resample video
    cap = cv2.VideoCapture(video_path)
    if cap.get(cv2.CAP_PROP_FPS) != VIDEO_FPS:
        if not os.path.exists(v_path):
            ffmpy.FFmpeg(
                inputs={video_path: None},
                outputs={v_path: '-qscale 0 -r ' + str(VIDEO_FPS) +
                                 ' -y -loglevel ' + LOG_LEVEL}
            ).run()
    else:
        v_path = video_path
    cap.release()

    return {'video_path': v_path, 'audio_path': w_path}


def upsample_video_feature(video_path, upsampling_rate):
    init_dde()
    low_rate_lm = landmark_feature(video_path, only_mouth=True)
    lm_feature = []
    for i in range(len(low_rate_lm) - 1):
        left = low_rate_lm[i]
        delta = low_rate_lm[i + 1] - left
        for k in range(upsampling_rate):
            lm_feature.append(left + delta * k / upsampling_rate)
    lm_feature.append(low_rate_lm[-1])  # the last one
    return np.asarray(lm_feature, np.float32)


def pre_process_video(video_path, audio_feature='mfcc',
                      winlen=0.025, winstep=0.01, **kwarg):
    # upsampling video feature
    video_path = os.path.abspath(video_path)
    video_path = video_path.replace('\\', '/')
    path = demux_video(video_path)

    upsampling_rate = 1 / (VIDEO_FPS * winstep)
    if upsampling_rate - int(upsampling_rate) != 0:
        raise ValueError('Upsampling rate is not integer.')
    upsampling_rate = int(upsampling_rate)
    print('Fetching video feature...')
    lm_feature = upsample_video_feature(path['video_path'], upsampling_rate)

    # audio feature
    print('Fetching audio feature...')
    print(path['audio_path'])
    rate, audio = wavfile.read(path['audio_path'])
    if audio_feature == 'mfcc':
        nfft = 512
        while winlen * rate > nfft:
            nfft = nfft << 1
        a_feature = mfcc(audio, rate, winlen=winlen,
                         winstep=winstep, nfft=nfft)
    elif audio_feature == 'lpc' or audio_feature == 'formant':
        # get k and pre_e from the kwarg
        k = kwarg['k'] if 'k' in kwarg else 4
        preemphsis = kwarg['pre_e'] if 'pre_e' in kwarg else None
        # calc lpc
        a_feature = lpc_feature(audio, rate,
                                winlen=winlen, winstep=winstep,
                                k=k, preemphsis=preemphsis)
        # extra step for formant
        if audio_feature == 'formant':
            for i in range(len(a_feature)):
                a_feature[i] = formant(a_feature[i], rate)
    else:
        raise NotImplementedError('No such audio feature!')

    # read align
    path_split = video_path.split('/')
    prefix = os.path.splitext(path_split[-1])[0]
    align_path = '/'.join(path_split[: -3]) + '/align/' + prefix + '.align'

    align = []
    if os.path.exists(align_path):
        with open(align_path) as file:
            for the_line in file:
                line = the_line.split(' ')
                start = int(line[0])
                end = int(line[1])
                word = line[2].strip()
                align.append({
                    'start': start,
                    'end': end,
                    'word': word
                })

    return {
        'video_feature': np.asarray(lm_feature, np.float32),
        'audio_feature': np.asarray(a_feature, np.float32),
        'alignment': align,
        'path': path
    }


# for sampling result

def mux(audio_path, video_path, media_path):
    ffmpy.FFmpeg(
        inputs={audio_path: None, video_path: None},
        outputs={media_path: '-loglevel ' + LOG_LEVEL}
    ).run()


def generate_video_from_landmarks(landmark_seq, video_path):
    global VIDEO_FPS
    global W
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (W, W))
    for i, frame in enumerate(landmark_seq):
        img = draw_mouth_landmarks(W, frame)
        out.write(img)
    out.release()


def generate_compare_video(pred_lms, true_lms, video_path):
    global VIDEO_FPS
    global W
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (W, W))
    for i in range(len(pred_lms)):
        frame = pred_lms[i]
        img = draw_mouth_landmarks(W, frame)
        frame = true_lms[i]
        img = draw_mouth_landmarks(W, frame, (0, 0, 255), img)
        out.write(img)
    out.release()


def sample_video(data, media_path):
    prefix, _ = os.path.splitext(media_path)
    if 'anime_pred' in data:
        m_path_pred = prefix + '_pred'
        generate_video_from_landmarks(data['anime_pred'], m_path_pred + '.avi')
        if os.path.exists(m_path_pred + '.mp4'):
            os.remove(m_path_pred + '.mp4')
        mux(
            data['source_dir'] + '.wav',
            m_path_pred + '.avi',
            m_path_pred + '.mp4'
        )
        os.remove(m_path_pred + '.avi')

    # true
    if 'anime_true' in data:
        m_path_true = prefix + '_true'
        generate_video_from_landmarks(data['anime_true'], m_path_true + '.avi')
        if os.path.exists(m_path_true + '.mp4'):
            os.remove(m_path_true + '.mp4')
        mux(
            data['source_dir'] + '.wav',
            m_path_true + '.avi',
            m_path_true + '.mp4'
        )
        os.remove(m_path_true + '.avi')


# the processed video must have even number of frames (paired for training)
# def generate_data(dir_path, E=24, loc=0.0, scale=0.01,
#                   audio_feature='lpc', winlen=0.016,
#                   winstep=0.008, **kwarg):
#     from utils.dir import find_files
#     lists = find_files(dir_path, 'mpg')
#     if 'k' in kwarg:
#         kwarg['k'] = int(kwarg['k'] / 2)
#     all_data = []
#     for file_path in lists:
#         data = pre_process_video(
#             file_path, audio_feature,
#             winlen, winstep, **kwarg
#         )
#         length = len(data['video_feature'])
#         if len(data['audio_feature']) < length:
#             length = len(data['audio_feature'])
#         length -= 10
#         if length & 1 > 0:
#             length -= 1
#         # random e vector
#         e_vector = np.random.normal(loc=loc, scale=scale, size=(E))
#         for i in range(length):
#             all_data.append({
#                 'input': data['audio_feature'][i],
#                 'output': data['video_feature'][i].flatten(),
#                 'e_vector': np.asarray(e_vector, np.float32)
#             })
#     return np.asarray(all_data)


if __name__ == '__main__':
    # from utils.dir import find_files
    # lists = find_files('Video', 'mpg')
    # for file_path in lists:
    #     print(pre_process_video(file_path, 'lpc')['audio_feature'][0])
    # print(generate_data('Video', k=32, pre_e=0.63)[-1]['e_vector'].shape)
    pass
