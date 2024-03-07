from wave import open
from pyaudio import PyAudio, paInt16
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import librosa
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import spectrogram
from scipy.fft import rfft

FRAMES_PER_BUFFER = 3200
FORMAT = paInt16
CHANNELS = 1


class WorkerSignals(QObject):
    update_progress = pyqtSignal(int)


class WorkerThread(QThread):
    def __init__(self):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()

    def run(self):
        pass


def record_audio(duration=2, sample_rate=44100):
    p = PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )
    print("Start recording...")

    seconds = duration
    frames = []
    end = int(sample_rate / FRAMES_PER_BUFFER * seconds)
    for i in range(0, end):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)
    print("end recording")

    stream.stop_stream()
    stream.close()
    save_path = "record.wav"
    obj = open(save_path, "wb")
    obj.setnchannels(CHANNELS)
    obj.setsampwidth(p.get_sample_size(FORMAT))
    obj.setframerate(sample_rate)
    obj.writeframes(b"".join(frames))
    obj.close()

    return save_path

def extract_passwords_features(path, audio_file=None, sr=None):
    if audio_file is None or sr is None:
        audio_file, sr = librosa.load(path)

    m = np.abs(librosa.stft(y=audio_file, n_fft=512))
    pca = PCA(n_components=10)  # Reduce to 2 dimensions for visualization
    pca_m = pca.fit_transform(m).flatten()

    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=100)
    mfccs_pca = pca.fit_transform(mfccs).flatten()

    # # chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=audio_file, sr=sr, n_chroma=100)
    chroma_stft_pca = pca.fit_transform(chroma_stft).flatten()

    combined_features = np.concatenate([pca_m, mfccs_pca, chroma_stft_pca])
    return combined_features


def extract_person_features(path, audio_file=None, sr=None):
    if audio_file is None or sr is None:
        audio_file, sr = librosa.load(path)

    pca = PCA(n_components=10)  # Reduce to 2 dimensions for visualization

    # abs of stft
    m = np.abs(librosa.stft(y=audio_file, n_fft=512))
    pca_m = pca.fit_transform(m).flatten()

    # mfccs
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr, n_mfcc=100)
    mfccs_pca = pca.fit_transform(mfccs).flatten()

    # Spectrogram
    f, t, Sxx = spectrogram(audio_file, sr)
    Sxx_pca = pca.fit_transform(Sxx).flatten()

    # # chroma_stft
    chroma_stft = librosa.feature.chroma_stft(y=audio_file, sr=sr, n_chroma=100)
    chroma_stft_pca = pca.fit_transform(chroma_stft).flatten()

    combined_features = np.concatenate([pca_m, mfccs_pca, Sxx_pca, chroma_stft_pca])

    return combined_features
