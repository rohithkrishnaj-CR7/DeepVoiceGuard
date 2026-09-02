"""
Synthetic Speech Generator & Robust Channel/Vocoder Artifact Simulator.
"""

import os
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import Tuple, List, Dict, Optional

class SyntheticAudioGenerator:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def apply_channel_and_codec(self, y: np.ndarray, channel_type: Optional[str] = None) -> np.ndarray:
        n_samples = len(y)
        if channel_type is None:
            channel_types = ["clean", "whatsapp_opus", "phone_amr", "room_mic", "mp3_lossy"]
            channel_type = np.random.choice(channel_types)

        out = np.copy(y)

        if channel_type == "whatsapp_opus":
            cutoff = np.random.uniform(5500.0, 7200.0)
            b, a = signal.butter(4, cutoff / (self.sr / 2.0), btype='low')
            out = signal.lfilter(b, a, out)
            codec_noise = np.random.randn(n_samples).astype(np.float32) * np.random.uniform(0.001, 0.004)
            out = out + codec_noise

        elif channel_type == "phone_amr":
            b, a = signal.butter(3, [300.0 / (self.sr / 2.0), 4800.0 / (self.sr / 2.0)], btype='band')
            out = signal.lfilter(b, a, out)

        elif channel_type == "room_mic":
            b, a = signal.butter(2, 120.0 / (self.sr / 2.0), btype='high')
            out = signal.lfilter(b, a, out)
            room_noise = np.random.randn(n_samples).astype(np.float32) * 0.003
            out = out + room_noise

        elif channel_type == "mp3_lossy":
            b, a = signal.butter(4, 5600.0 / (self.sr / 2.0), btype='low')
            out = signal.lfilter(b, a, out)

        m = np.max(np.abs(out))
        if m > 1e-5:
            out = out / m * 0.85
        return out.astype(np.float32)

    def generate_human_like_voice(
        self,
        duration: float = 3.0,
        base_f0: float = 130.0,
        apply_codec: bool = True
    ) -> np.ndarray:
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Natural human prosody contour (f0 variation 15-35 Hz across phrases) + organic jitter
        prosody = 1.0 + 0.18 * np.sin(2 * np.pi * 1.1 * t) + 0.09 * np.sin(2 * np.pi * 0.4 * t + 0.8)
        jitter = 1.0 + np.random.uniform(0.015, 0.035) * np.random.randn(n_samples)
        f0_curve = base_f0 * prosody * jitter
        f0_curve = np.clip(f0_curve, 75.0, 360.0)

        phase = 2 * np.pi * np.cumsum(f0_curve) / self.sr
        excitation = np.zeros(n_samples, dtype=np.float32)

        for h in range(1, 13):
            shimmer = 1.0 + 0.04 * np.random.randn(n_samples)
            harmonic_amp = (1.0 / (h ** 1.15)) * shimmer
            excitation += (harmonic_amp * np.sin(h * phase)).astype(np.float32)

        # Dynamic vowel formants
        vowel_blend = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
        f1 = 450.0 + 300.0 * vowel_blend
        f2 = 1100.0 + 900.0 * (1.0 - vowel_blend)

        filtered = np.zeros_like(excitation)
        b1, a1 = signal.iirpeak(np.mean(f1) / (self.sr / 2.0), 6.0)
        filtered += 1.0 * signal.lfilter(b1, a1, excitation)
        b2, a2 = signal.iirpeak(np.mean(f2) / (self.sr / 2.0), 8.0)
        filtered += 0.6 * signal.lfilter(b2, a2, excitation)
        b3, a3 = signal.iirpeak(2600.0 / (self.sr / 2.0), 12.0)
        filtered += 0.3 * signal.lfilter(b3, a3, excitation)

        breath = np.random.randn(n_samples).astype(np.float32) * 0.02
        b_noise, a_noise = signal.butter(3, 1000.0 / (self.sr / 2.0), btype='high')
        filtered += signal.lfilter(b_noise, a_noise, breath)

        m = np.max(np.abs(filtered))
        if m > 1e-6:
            filtered = filtered / m * 0.85

        if apply_codec:
            filtered = self.apply_channel_and_codec(filtered)

        return filtered.astype(np.float32)

    def generate_cloned_voice_artifacts(
        self,
        base_audio: Optional[np.ndarray] = None,
        duration: float = 3.0,
        artifact_type: str = "neural_vocoder",
        apply_codec: bool = True
    ) -> np.ndarray:
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        if artifact_type == "neural_vocoder":
            # AI Neural Vocoder: Robotic static pitch + diffusion checkerboard noise + high-freq comb
            f0_static = 140.0
            phase = 2 * np.pi * f0_static * t  # Flat robotic pitch!
            cloned = np.zeros(n_samples, dtype=np.float32)
            for h in range(1, 10):
                cloned += (1.0 / (h ** 1.1)) * np.sin(h * phase)
            
            # Vocoder formant
            b, a = signal.iirpeak(900.0 / (self.sr / 2.0), 6.0)
            cloned = signal.lfilter(b, a, cloned)
            
            # Checkerboard sub-band noise
            diff_noise = np.random.randn(n_samples).astype(np.float32) * 0.04
            b_band, a_band = signal.butter(3, [2000.0 / (self.sr / 2.0), 5000.0 / (self.sr / 2.0)], btype='band')
            cloned += signal.lfilter(b_band, a_band, diff_noise)

        elif artifact_type == "griffin_lim":
            # Broken glottal phase
            f0_static = 185.0
            phase = 2 * np.pi * f0_static * t
            cloned = np.zeros(n_samples, dtype=np.float32)
            for h in range(1, 8):
                cloned += (1.0 / (h ** 1.1)) * np.sin(h * phase)

            stft = np.fft.rfft(cloned)
            random_phase = np.exp(1j * np.random.uniform(-1.2, 1.2, len(stft)))
            cloned = np.fft.irfft(stft * random_phase, n=n_samples)
            b, a = signal.iirpeak(2400.0 / (self.sr / 2.0), 20.0)
            cloned += 0.4 * signal.lfilter(b, a, cloned)

        else: # voice_conversion
            f0_step = 130.0 + 35.0 * np.heaviside(t - duration / 2.0, 1.0) # Unnatural step jump
            phase = 2 * np.pi * np.cumsum(f0_step) / self.sr
            cloned = np.zeros(n_samples, dtype=np.float32)
            for h in range(1, 9):
                cloned += (1.0 / (h ** 1.15)) * np.sin(h * phase)
            b_warp, a_warp = signal.iirpeak(1600.0 / (self.sr / 2.0), 8.0)
            cloned += 0.4 * signal.lfilter(b_warp, a_warp, cloned)

        m = np.max(np.abs(cloned))
        if m > 1e-6:
            cloned = cloned / m * 0.85

        if apply_codec:
            cloned = self.apply_channel_and_codec(cloned)

        return cloned.astype(np.float32)

    def generate_demo_samples(
        self,
        output_dir_real: str = "sample_data/real",
        output_dir_cloned: str = "sample_data/cloned"
    ) -> Dict[str, List[str]]:
        os.makedirs(output_dir_real, exist_ok=True)
        os.makedirs(output_dir_cloned, exist_ok=True)

        real_files = []
        cloned_files = []

        real_configs = [
            ("sample_genuine_male_voice_1.wav", 115.0, 3.5),
            ("sample_genuine_female_voice_2.wav", 215.0, 3.2),
            ("sample_genuine_speech_natural_3.wav", 145.0, 4.0),
        ]
        for fname, f0, dur in real_configs:
            path = os.path.join(output_dir_real, fname)
            audio = self.generate_human_like_voice(duration=dur, base_f0=f0, apply_codec=False)
            sf.write(path, audio, self.sr)
            real_files.append(path)

        cloned_configs = [
            ("sample_cloned_elevenlabs_ai_1.wav", "neural_vocoder", 130.0, 3.5),
            ("sample_cloned_xtts_synthetic_2.wav", "griffin_lim", 190.0, 3.2),
            ("sample_cloned_voice_conversion_3.wav", "voice_conversion", 150.0, 4.0),
        ]
        for fname, art_type, f0, dur in cloned_configs:
            path = os.path.join(output_dir_cloned, fname)
            audio = self.generate_cloned_voice_artifacts(duration=dur, artifact_type=art_type, apply_codec=False)
            sf.write(path, audio, self.sr)
            cloned_files.append(path)

        return {'real': real_files, 'cloned': cloned_files}
