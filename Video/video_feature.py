import os
import sys
import cv2
import dde
import numpy
import fbxanime

DDE_PATH = 'C:/Users/yushroom/AppData/Local/conda/conda/envs/tensorflow/Lib/site-packages/v3.bin'
FBX_PATH = 'C:/Users/yushroom/AppData/Local/conda/conda/envs/tensorflow/Lib/site-packages/fbx_anime.fbx'

g_dde_inited = False

def init_dde_fbx(
        dde_path=DDE_PATH,
        fbx_path=FBX_PATH,
        fbx_w=1280, fbx_h=720):

    if os.path.exists(dde_path) and os.path.exists(fbx_path):
        dde.init(dde_path)
        dde.set_n_copies(120)
        fbxanime.init(fbx_w, fbx_h, fbx_path)


def init_dde(dde_path=DDE_PATH):
    global g_dde_inited
    if not g_dde_inited:
        dde.init(dde_path)
        dde.set_n_copies(120)
        g_dde_inited = True


def draw_line(zeros, points, delta=0):
    for i in range(46 + delta, 57 + delta):
        cv2.line(zeros, points[i], points[i + 1], (150, 150, 150), 1)
    for i in range(58 + delta, 60 + delta):
        cv2.line(zeros, points[i], points[i + 1], (150, 150, 150), 1)
    for i in range(61 + delta, 63 + delta):
        cv2.line(zeros, points[i], points[i + 1], (150, 150, 150), 1)
    for i, j in zip([46 + delta, 46 + delta, 46 + delta, 52 + delta, 52 + delta],
                    [57 + delta, 58 + delta, 63 + delta, 60 + delta, 61 + delta]):
        cv2.line(zeros, points[i], points[j], (150, 150, 150), 1)


def draw_mouth_landmarks(W, lm, color=(0, 255, 0), img=None, delta=(0, 0)):
    if img is None:
        img = numpy.zeros((W, W, 3), dtype=numpy.uint8)
    points = []
    for i in range(len(lm)):
        x = int(lm[i][0] + W / 2) + delta[0]
        y = int(lm[i][1] + W / 2) + delta[1]
        points.append((x, y))
        cv2.circle(img, (x, y), 1, color, 2)
    draw_line(img, points, -46)
    return img


def show_all_landmarks(img):
    W = 800
    result = dde.get("landmarks_ar")
    min = numpy.min(result)
    result -= min
    result *= 2
    mean_x, mean_y = result[::3].mean(), result[1::3].mean()
    print(mean_x, mean_y)

    zeros = numpy.zeros((W, W, 3), dtype=numpy.uint8)
    points = []
    for i in range(75):
        x = 800 - int(result[i * 3] - mean_x + 400)
        y = 800 - int(result[i * 3 + 1] - mean_y + 400)
        points.append((x, y))
        cv2.circle(zeros, (x, y), 1, (0, 255, 0), 2)
    draw_line(zeros, points)
    cv2.imshow("landmarks_ar", zeros)

    tmp = img
    result = dde.get("landmarks")
    points = []
    for i in range(75):
        points.append((result[i * 2], result[i * 2 + 1]))
        cv2.circle(tmp, (result[i * 2], result[i * 2 + 1]), 1, (0, 255, 0), 2)
    draw_line(tmp, points)
    cv2.imshow("landmarks", tmp)
    cv2.waitKey(30)


def landmarks_ar():
    result = dde.get("landmarks_ar")
    min = numpy.min(result)
    result -= min
    result *= 2
    mean_x, mean_y = result[::3].mean(), result[1::3].mean()

    lm = []
    for i in range(75):
        lm.append((mean_x - result[i * 3], mean_y - result[i * 3 + 1]))
    return lm


def noop_scan(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret = False
        for _ in range(200):
            ret |= dde.run(frame)
            if ret:
                break
    cap.release()


def scan_media(video_path, try_limit=10, repeat=30, show=False):
    noop_scan(video_path)
    cap = cv2.VideoCapture(video_path)

    anime_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # get landmarks
        ret = False
        for _ in range(try_limit):
            ret |= dde.run(frame)
            if ret:
                break
        if not ret:
            cap.release()
            return None

        for _ in range(repeat):
            dde.run(frame)

        # push anime data
        lm = landmarks_ar()
        anime_data.append(numpy.asarray(lm, dtype=numpy.float32))
        if show:
            show_all_landmarks(frame)
            # img = draw_mouth_landmarks(800, numpy.array(lm)[46: 64])
            # cv2.imshow('frame', img)
            # cv2.waitKey(30)

    cap.release()
    return anime_data


def landmark_feature(video_path, only_mouth=False, force_clean=False, try_limit=10, repeat=30, show=False):
    prefix, _ = os.path.splitext(video_path)
    lm_path = prefix + '.lm'
    need_scan = True
    if os.path.exists(lm_path):
        if force_clean:
            os.remove(lm_path)
        else:
            need_scan = False
    if need_scan:
        print('Scaning media', video_path)
        anime_data = scan_media(video_path, try_limit, repeat, show)
        if anime_data is None:
            os.remove(video_path)
            return None
        with open(lm_path, 'w') as file:
            for data in anime_data:
                for d in data:
                    file.write(str(d[0]) + ' ' + str(d[1]) + ' ')
                file.write('\n')
    else:
        # get from cache
        print('Getting from cache...')
        anime_data = []
        with open(lm_path) as file:
            for line in file:
                lms = line.strip().split(' ')
                t = []
                for i in range(int(len(lms) / 2)):
                    t.append((float(lms[i * 2]), float(lms[i * 2 + 1])))
                assert(len(t) == 75)
                t = numpy.asarray(t, dtype=numpy.float32)
                anime_data.append(t)
    if only_mouth:
        for i, data in enumerate(anime_data):
            anime_data[i] = data[46: 64]
    return anime_data


if __name__ == '__main__':
    # find files
    def find_files(path, target_ext):
        if target_ext[0] != '.':
            target_ext = '.' + target_ext
        result_list = []
        for parent, dirs, files in os.walk(path):
            for file in files:
                name, ext = os.path.splitext(os.path.join(parent, file))
                if ext == target_ext:
                    result_list.append(name + ext)
        return result_list

    init_dde()
    lists = find_files('GRID', 'mpg')
    length = len(lists)
    for i, video_file in enumerate(lists):
        print('[' + str(i) + '/' + str(length) + ']')
        landmark_feature(video_file)


# if __name__ == '__main__':
#     init_dde()
#     # if len(sys.argv) == 1:
#     #     scan_media('bbieza.mpg', show=True)
#     # else:
#     #     scan_media(sys.argv[1], show=True)
#     anime = landmark_feature('bbieza.mpg', only_mouth=True)
#     for data in anime:
#         img = draw_mouth_landmarks(800, data)
#         cv2.imshow('frame', img)
#         cv2.waitKey(30)
