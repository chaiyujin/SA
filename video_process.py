import os
import cv2
import dde
import ffmpy
from audio_feature import *
from video_feature import *

W = 800
VIDEO_FPS = 25.0
DDE_PATH = 'C:/Users/yushroom/Anaconda2/envs/tensorflow/Lib/site-packages/v3.bin'

# for generating data

def demux_video(video_path, clear_old=False):
    global VIDEO_FPS

    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        return None
    file_prefix, _ = os.path.splitext(video_path)
    print('Demuxing', video_path)
    v_path = file_prefix + '_video.mp4'
    a_path = file_prefix + '_audio.mp4'
    w_path = file_prefix + '_audio.wav'
    if clear_old:
        remove_files([v_path, a_path, w_path])

    # 1. demux audio
    if not os.path.exists(w_path):
        ffmpy.FFmpeg(
            inputs={video_path: None},
            outputs={
                a_path: '-map 0:1 -c:a copy -f mp4 -loglevel ' + loglevel}
        ).run()
        # rms normalize the audio
        true, w_path = rms_normalize(a_path)
        if not true:
            return None
        # delete middle result
        os.remove(a_path)
    # 2. resample video
    cap = cv2.VideoCapture(video_path)
    if cap.get(cv2.CAP_PROP_FPS) != VIDEO_FPS:
        if not os.path.exists(v_path):
            ffmpy.FFmpeg(
                inputs={video_path: None},
                outputs={v_path: '-qscale 0 -r ' + str(VIDEO_FPS) +
                                 ' -y -loglevel ' + loglevel}
            ).run()
    else:
        v_path = video_path
    cap.release()

    return {'video_path': v_path, 'audio_path': w_path}


# for sampling result

def mux(audio_path, video_path, media_path):
    ffmpy.FFmpeg(
        inputs={audio_path: None, video_path: None},
        outputs={media_path: '-loglevel ' + loglevel}
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
