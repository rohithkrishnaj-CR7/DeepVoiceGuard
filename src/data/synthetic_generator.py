"""
Synthetic Speech Generator & Artifact Simulator.
Generates procedural speech signals with:
1. Genuine biological vocal cord & formant dynamics.
2. Cloned / Neural TTS synthetic artifacts (phase distortion, robotic pitch flattening,
   vocoder high-frequency cutoff, harmonic noise injection, Griffin-Lim artifacts).
Used to synthesize benchmark training datasets and ready-to-test demo samples.
"""

import os
import numpy as np
import scipy.signal as signal
import soundfile as sf
from typing import Tuple, List, Dict

class SyntheticAudioGenerator:
    def __init__(self, sr: int = 16000):
        self.sr = sr

    def generate_human_like_voice(
        self,
        duration: float = 3.0,
        base_f0: float = 130.0,
        phoneme_sequence: Optional[List[Tuple[float, float, float]]] = None
    ) -> np.ndarray:
        """
        Synthesizes human-like voice with natural F0 jitter, shimmer, breathing noise,
        and physiological formant resonances (F1, F2, F3).
        """
        n_samples = int(duration * self.sr)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # 1. Natural pitch modulation (prosody contour + micro-jitter)
        prosody = 1.0 + 0.12 * np.sin(2 * np.pi * 1.5 * t) + 0.05 * np.sin(2 * np.pi * 0.4 * t)
        jitter = 1.0 + 0.02 * np.random.randn(n_samples)
        f0_curve = base_f0 * prosody * jitter
        f0_curve = np.clip(f0_curve, 70.0, 350.0)

        # 2. Glottal excitation pulse (Rosenberg-style model + harmonics)
        phase = 2 * np.pi * np.cumsum(f0_curve) / self.sr
        excitation = np.zeros(n_samples, dtype=np.float32)
        for h in range(1, 12):
            harmonic_amp = (1.0 / (h ** 1.3)) * (1.0 + 0.03 * np.random.randn(n_samples))
            excitation += (harmonic_amp * np.sin(h * phase)).astype(np.float32)

        # 3. Formant resonant filtering (Vocal tract modeling)
        # Default vowel /a/ or sequence: F1 ~ 750Hz, F2 ~ 1250Hz, F3 ~ 2600Hz
        formants = [(750.0, 80.0, 1.0), (1250.0, 110.0, 0.7), (2600.0, 150.0, 0.4), (3500.0, 200.0, 0.2)]
        filtered = np.zeros_like(excitation)

        for f_center, bw, gain in formants:
            b, a = signal.iirpeak(f_center / (self.sr / 2.0), f_center / bw)
            filtered += (gain * signal.lfilter(b, a, excitation)).astype(np.float32)

        # 4. Organic aspiration and breathing noise
        breath = np.random.randn(n_samples).astype(np.float32) * 0.03
        b_noise, a_noise = signal.butter(3, 800.0 / (self.sr / 2.0), btype='high')
        filtered += signal.lfilter(b_noise, a_noise, breath)

        # 5. Peak normalization
        max_val = np.max(np.abs(filtered))
        if max_val > 1e-6:
            filtered = filtered / max_val * 0.85

        return filtered.astype(np.float32)

    def generate_cloned_voice_artifacts(
        self,
        base_audio: Optional[np.ndarray] = None,
        duration: float = 3.0,
        artifact_type: str = "neural_vocoder"
    ) -> np.ndarray:
        """
        Applies distinctive synthetic artifacts to speech:
        - "neural_vocoder": High-frequency cutoff, robotic pitch flattening, diffusion noise
        - "griffin_lim": Phase cancellation, metallic ringing
        - "voice_conversion": Formant warping and unnatural pitch jumps
        """
        if base_audio is None:
            base_audio = self.generate_human_like_voice(duration=duration, base_f0=140.0)

        cloned = np.copy(base_audio)
        n_samples = len(cloned)

        if artifact_type == "neural_vocoder":
            # 1. High frequency spectral truncation (>6.5 kHz sharp cutoff)
            b, a = signal.butter(6, 6200.0 / (self.sr / 2.0), btype='low')
            cloned = signal.lfilter(b, a, cloned)

            # 2. Robotic pitch flattening: modulate with synthetic harmonic carrier
            t = np.linspace(0, duration, n_samples, endpoint=False)
            carrier = np.sin(2 * np.pi * 140.0 * t)
            cloned = 0.80 * cloned + 0.20 * (cloned * carrier)

            # 3. Vocoder checkerboard / diffusion noise in mid-high bands
            diff_noise = np.random.randn(n_samples).astype(np.float32) * 0.02
            b_band, a_band = signal.butter(3, [2000.0 / (self.sr / 2.0), 5000.0 / (self.sr / 2.0)], btype='band')
            cloned += signal.lfilter(b_band, a_band, diff_noise)

        elif artifact_type == "griffin_lim":
            # Phase scrambling artifact
            stft = np.fft.rfft(cloned)
            random_phase = np.exp(1j * np.random.uniform(-0.6, 0.6, len(stft)))
            cloned = np.fft.irfft(stft * random_phase, n=n_samples)

            # Metallic resonance filter
            b, a = signal.iirpeak(2400.0 / (self.sr / 2.0), 30.0)
            cloned += 0.3 * signal.lfilter(b, a, cloned)

        elif artifact_type == "voice_conversion":
            # Formant warping and unnatural pitch stepping
            b_warp, a_warp = signal.iirpeak(1600.0 / (self.sr / 2.0), 10.0)
            cloned += 0.4 * signal.lfilter(b_warp, a_warp, cloned)
            # Sudden pitch step jump in middle
            mid = n_samples // 2
            t_mid = np.linspace(0, duration / 2.0, n_samples - mid)
            cloned[mid:] = cloned[mid:] * np.cos(2 * np.pi * 45.0 * t_mid)

        # Peak normalization
        max_val = np.max(np.abs(cloned))
        if max_val > 1e-6:
            cloned = cloned / max_val * 0.85

        return cloned.astype(np.float32)

    def generate_demo_samples(
        self,
        output_dir_real: str = "sample_data/real",
        output_dir_cloned: str = "sample_data/cloned"
    ) -> Dict[str, List[str]]:
        """
        Creates ready-to-test sample audio files for genuine and cloned voices.
        """
        os.makedirs(output_dir_real, exist_ok=True)
        os.makedirs(output_dir_cloned, exist_ok=True)

        real_files = []
        cloned_files = []

        # Generate Real Samples
        real_configs = [
            ("sample_genuine_male_voice_1.wav", 110.0, 3.5),
            ("sample_genuine_female_voice_2.wav", 210.0, 3.2),
            ("sample_genuine_speech_natural_3.wav", 145.0, 4.0),
        ]
        for fname, f0, dur in real_configs:
            path = os.path.join(output_dir_real, fname)
            audio = self.generate_human_like_voice(duration=dur, base_f0=f0)
            sf.write(path, audio, self.sr)
            real_files.append(path)

        # Generate Cloned Samples
        cloned_configs = [
            ("sample_cloned_elevenlabs_ai_1.wav", "neural_vocoder", 130.0, 3.5),
            ("sample_cloned_xtts_synthetic_2.wav", "griffin_lim", 190.0, 3.2),
            ("sample_cloned_voice_conversion_3.wav", "voice_conversion", 150.0, 4.0),
        ]
        for fname, art_type, f0, dur in cloned_configs:
            path = os.path.join(output_dir_cloned, fname)
            base = self.generate_human_like_voice(duration=dur, base_f0=f0)
            cloned_audio = self.generate_cloned_voice_artifacts(base, duration=dur, artifact_type=art_type)
            sf.write(path, cloned_audio, self.sr)
            cloned_files.append(path)

        print(f"Generated {len(real_files)} genuine and {len(cloned_files)} cloned demo sample files.")
        return {'real': real_files, 'cloned': cloned_files}
