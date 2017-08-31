import os
import math
import ffmpy
import subprocess
import numpy as np
import scipy.io.wavfile
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.fftpack import dct
from scipy.signal import lfilter, hamming
from python_speech_features import fbank, delta, lifter


def rms_normalize(input_audio):
    input_audio = input_audio.replace('\\', '/')
    cmd = ['ffmpeg-normalize', '-v', '-f', '-e', '-ar 25000 -ac 1',
           input_audio]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    p.communicate()
    if p.returncode == 0:
        full_prefix = os.path.splitext(input_audio)[0]
        file_prefix = full_prefix.split('/')
        path_prefix = '/'.join(file_prefix[0: -1])
        file_prefix = file_prefix[-1]
        file_name = 'normalized-' + file_prefix + '.wav'
        if len(path_prefix) > 0:
            file_name = path_prefix + '/' + file_name
        if not os.path.exists(file_name):
            ffmpy.FFmpeg(
                inputs={input_audio: None},
                outputs={file_name: '-f wav -vn -ac 1 -ar 25000 -y -loglevel error'}
            ).run()
        return True, file_name
    else:
        return False, ''


def draw_spec(spec, T, A, aspect):
    plt.figure(figsize=(15, 8))
    plt.imshow(spec, cmap=cm.jet, aspect=aspect, extent=[0, T, 0, A])
    plt.show()


def audio_feature(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                  numcep=13, nfilt=40, nfft=512, lowfreq=0, highfreq=None,
                  preemph=0.97, ceplifter=22, appendEnergy=True,
                  winfunc=np.hamming):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, preemph, winfunc)
    log_fbank = np.log(feat)
    # discard the 0-th dct coefficient
    mfcc = dct(log_fbank, type=2, axis=1, norm='ortho')[:, 1:numcep]
    mfcc = lifter(mfcc, ceplifter)
    d1_mfcc = delta(mfcc, 1)
    d2_mfcc = delta(d1_mfcc, 1)
    energy = np.reshape(np.log(energy), (energy.shape[0], 1))
    mixed = np.concatenate((mfcc, d1_mfcc, d2_mfcc, energy), axis=1)
    return mixed


def mfcc_energy(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                numcep=13, nfilt=40, nfft=512, lowfreq=0, highfreq=None,
                preemph=0.97, ceplifter=22, appendEnergy=True,
                winfunc=np.hamming):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, preemph, winfunc)
    log_fbank = np.log(feat)
    # discard the 0-th dct coefficient
    mfcc = dct(log_fbank, type=2, axis=1, norm='ortho')[:, 1:numcep]
    mfcc = lifter(mfcc, ceplifter)
    energy = np.reshape(np.log(energy), (energy.shape[0], 1))
    return mfcc, energy


def draw_wav(fs, signal):
    time_seq = np.linspace(0, len(signal)/fs, num=len(signal))
    plt.figure(figsize=(10, 3))
    plt.title("Wav")
    plt.subplot(111)
    plt.grid(color='gray', linestyle='--', linewidth=0.2)
    plt.plot(time_seq, signal)
    plt.show()


def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.

    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:

      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]

    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.

    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)"""

    order = int(order)

    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size+order]
        phi = np.dot(linalg.inv(linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi)), None, None
    else:
        return np.ones(1, dtype=signal.dtype), None, None


def lpc_coefficient(signal, k=4, preemphsis=0.63):
    x1 = signal * hamming(len(signal))
    if preemphsis is not None:
        x1 = lfilter([1], [1., preemphsis], x1)
    ncoeff = k * 2
    try:
        A, e, k = lpc(x1, ncoeff)
    except:
        A = np.zeros((ncoeff + 1), np.float32)
    # print(A.shape)
    return A


def formant(lpc_coefficient, samplerate=16000):
    # Get roots.
    rts = np.roots(lpc_coefficient)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    frqs = sorted(angz * (samplerate / (2 * math.pi)))
    return frqs


def lpc_feature(signal, samplerate=16000, winlen=0.025, winstep=0.01,
                k=4, preemphsis=None, winfunc=np.hamming):
    # 1. framing
    frame_length, frame_step = winlen * samplerate, winstep * samplerate
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    signal_length = len(signal)
    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
    )

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = \
        np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
        np.tile(np.arange(0, num_frames * frame_step, frame_step),
                (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 2. Window
    frames *= np.hamming(frame_length)

    ret = []
    for i in range(len(frames)):
        ret.append(lpc_coefficient(frames[i], k, preemphsis))

    return ret


if __name__ == '__main__':
    # sample of mfcc
    successful, norm_file = rms_normalize('pwwq9s.mpg')
    print(successful, norm_file)
    if successful:
        sample_rate, signal = scipy.io.wavfile.read(norm_file)
        signal = signal[0:int(3.5 * sample_rate)]
        draw_wav(sample_rate, signal)
        print(sample_rate)
        signal = signal[0:int(3.5 * sample_rate)]
        feat = audio_feature(signal, sample_rate)
        draw_spec(np.flipud(feat.T), 3.5, 37, 0.05)

    # sample of lpc feature
    sample_rate, signal = scipy.io.wavfile.read('normalized-bbie8s.wav')
    print(lpc_feature(signal, sample_rate, k=4, preemphsis=0.63)[:, 1: 3])
