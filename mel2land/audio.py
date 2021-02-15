from scipy import signal
import librosa
import numpy as np

min_level_db=-100
ref_level_db=20
signal_normalization = True
allow_clipping_in_normalization=True
symmetric_mels=True
max_abs_value=4.
fmin=55,
	# Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To
	# test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
fmax=7600,

_mel_basis = None

def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)

def _build_mel_basis():
    return mel(16000,1024,n_mels=80,fmin=fmin, fmax=fmax)

def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav

def _amp_to_db(x):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                           -max_abs_value, max_abs_value)
        else:
            return np.clip(max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value)

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))

def melspectrogram(wav):
    D = _stft(preemphasis(wav,0.97, True))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - ref_level_db

    if signal_normalization:
        S = _normalize(S)
        return S.T.astype(np.float32)
    return S.T.astype(np.float32)

def _stft(audio_in,n_fft=1024,win_length=1024,hop_length=256):
    return librosa.stft(audio_in, n_fft=n_fft, win_length=win_length, hop_length=hop_length)


def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm='slaney', dtype=np.float32):

    if fmax is None:
        fmax = float(sr) / 2


    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

    # Center freqs of each FFT bin
    fftfreqs = librosa.filters.fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = librosa.filters.mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    mel_f = np.squeeze(mel_f, axis=1)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm in (1, 'slaney'):
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]



    return weights