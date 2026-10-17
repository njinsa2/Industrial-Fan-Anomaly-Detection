import librosa
import numpy as np
from pathlib import Path

def mel_representation(
    file_path: Path, 
    n_mels: int = 128, 
    sr: int = 16000, 
    vector_form: bool = True, 
    descriptors: bool = True
) -> np.ndarray:
    #load audio using the passed sample rate
    sound_sample, sr = librosa.load(file_path, sr=sr)

    #compute mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=sound_sample, 
        sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=n_mels
    )

    #convert to log scale in dB
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    if vector_form:
        if descriptors:
            #static spectral statistics across time
            mel_means = np.mean(mel_spectrogram_db, axis=1)
            mel_stds = np.std(mel_spectrogram_db, axis=1)
            mel_max = np.max(mel_spectrogram_db, axis=1)

            #dynamic temporal derivatives (velocity and acceleration of sound change)
            delta = librosa.feature.delta(mel_spectrogram_db)
            delta2 = librosa.feature.delta(mel_spectrogram_db, order=2)
            
            delta_mean = np.mean(np.abs(delta), axis=1)
            delta2_mean = np.mean(np.abs(delta2), axis=1)

            #global bioacoustic instability descriptors (mean and std across frames)
            zcr = librosa.feature.zero_crossing_rate(sound_sample)[0]
            rms = librosa.feature.rms(y=sound_sample)[0]
            centroid = librosa.feature.spectral_centroid(y=sound_sample, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=sound_sample, sr=sr)[0]
            flatness = librosa.feature.spectral_flatness(y=sound_sample)[0]

            global_stats = np.array([
                np.mean(zcr), np.std(zcr),
                np.mean(rms), np.std(rms),
                np.mean(centroid), np.std(centroid),
                np.mean(rolloff), np.std(rolloff),
                np.mean(flatness), np.std(flatness)
            ])

            #total feature length = (5 * n_mels) + 10
            transformed_sample = np.concatenate([
                mel_means, 
                mel_stds, 
                mel_max, 
                delta_mean, 
                delta2_mean, 
                global_stats
            ])
        else:
            transformed_sample = mel_spectrogram_db.flatten()

        return transformed_sample

    return mel_spectrogram_db

