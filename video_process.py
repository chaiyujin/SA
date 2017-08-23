from __future__ import division
import os
import cv2
import ffmpy
from scipy.io import wavfile
from python_speech_features import mfcc
from Audio.feature.audio_feature import *
# from Video.video_feature import *

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

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return None
    file_prefix, _ = os.path.splitext(video_path)
    print('Demuxing', video_path)
    v_path = file_prefix + '_video.mp4'
    w_path = file_prefix + '_audio.wav'
    if clear_old:
        remove_files([v_path, w_path])

    # 1. demux audio
    if not os.path.exists(w_path):
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


def pre_process_video(video_path):
    # upsampling video feature
    path = demux_video(video_path)
    # low_rate_lm = landmarks_feature(path['video_path'], only_mouth=True)
    # lm_feature = []
    # for i in range(len(low_rate_lm) - 1):
    #     left = low_rate_lm[i]
    #     delta = low_rate_lm[i + 1] - left
    #     for k in range(4):
    #         lm_feature.append(left + delta * k / 4)

    # audio feature
    rate, audio = wavfile.read(path['audio_path'])
    mfcc_feature = mfcc(audio, rate, nfft=1024)

    # read align
    video_path = os.path.abspath(video_path)
    path_split = video_path.split('/')
    prefix = os.path.splitext(path_split[-1])[0]
    align_path = '/'.join(path_split[: -2]) + '/align/' + prefix + '.align'
    print(video_path)
    print(align_path)

# for sampling result

def mux(audio_path, video_path, media_path):
    ffmpy.FFmpeg(
        inputs={audio_path: None, video_path: None},
        outputs={media_path: '-loglevel ' + LOG_LEVEL}
    ).run()


# def generate_video_from_landmarks(landmark_seq, video_path):
#     global VIDEO_FPS
#     global W
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (W, W))
#     for i, frame in enumerate(landmark_seq):
#         img = draw_mouth_landmarks(W, frame)
#         out.write(img)
#     out.release()


# def sample_video(data, media_path):
#     prefix, _ = os.path.splitext(media_path)
#     if 'anime_pred' in data:
#         m_path_pred = prefix + '_pred'
#         generate_video_from_landmarks(data['anime_pred'], m_path_pred + '.avi')
#         if os.path.exists(m_path_pred + '.mp4'):
#             os.remove(m_path_pred + '.mp4')
#         mux(
#             data['source_dir'] + '.wav',
#             m_path_pred + '.avi',
#             m_path_pred + '.mp4'
#         )
#         os.remove(m_path_pred + '.avi')

#     # true
#     if 'anime_true' in data:
#         m_path_true = prefix + '_true'
#         generate_video_from_landmarks(data['anime_true'], m_path_true + '.avi')
#         if os.path.exists(m_path_true + '.mp4'):
#             os.remove(m_path_true + '.mp4')
#         mux(
#             data['source_dir'] + '.wav',
#             m_path_true + '.avi',
#             m_path_true + '.mp4'
#         )
#         os.remove(m_path_true + '.avi')

if __name__ == '__main__':
    pre_process_video('bbie8n.mpg')
