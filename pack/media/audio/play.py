import pyaudio
import wave
import time
import scipy.io.wavfile as wav
from feature.audio_feature import rms_normalize

chunk = 1024

_, wav_path = rms_normalize('bbieza.mpg')
rate, audio = wav.read('test.wav')
print(rate)
audio_idx = 0
# audio = audio[16000: 21250]

def block_play():
    p = pyaudio.PyAudio()

    stream = p.open(format = p.get_format_from_width(2),
                    channels = 1,
                    rate = rate,
                    output = True)

    audio_idx = 0
    while True:
        data = audio[audio_idx * chunk: (audio_idx + 1) * chunk].tobytes()
        if len(data) == 0: break
        stream.write(data)
        audio_idx += 1

    stream.close()
    p.terminate()


def callback_play():
    p = pyaudio.PyAudio()
    global audio_idx
    audio_idx = 0
    def callback(in_data, frame_count, time_info, status):
        global audio_idx
        data = audio[audio_idx: audio_idx + frame_count].tobytes()
        audio_idx += frame_count
        return (data, pyaudio.paContinue)
    stream = p.open(format = p.get_format_from_width(2),
                    channels = 1,
                    rate = rate,
                    output = True,
                    stream_callback=callback)

    stream.start_stream()
    while stream.is_active():
        time.sleep(0.1)
    stream.stop_stream()

    stream.close()
    p.terminate()

print(audio.shape)
print(rate)
# block_play()
callback_play()
