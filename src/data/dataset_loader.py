"""
Dataset Loader & Batch Feature Preprocessing Utility.
Loads audio from directories or CSV protocols, extracts feature matrices,
and formats data for Tabular and Deep Learning models.
"""

import os
import glob
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from ..features.audio_loader import AudioLoader, DEFAULT_SAMPLE_RATE
from ..features.extractor import FeatureExtractor
from .synthetic_generator import SyntheticAudioGenerator

class VoiceDatasetLoader:
    def __init__(self, sr: int = DEFAULT_SAMPLE_RATE):
        self.sr = sr
        self.audio_loader = AudioLoader(target_sr=sr)
        self.extractor = FeatureExtractor(sr=sr)

    def load_from_directory(
        self,
        real_dir: str,
        cloned_dir: str,
        max_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Scans real and cloned audio directories and extracts:
        - X_tab: (N, n_features) tabular feature matrix
        - X_mel: (N, 1, 128, 188) Log-Mel spectrograms
        - X_lfcc: (N, 1, 60, 300) LFCC tensors
        - y: (N,) binary ground truth labels (0=Real, 1=Cloned)
        - feature_names: List of tabular feature names
        """
        real_files = glob.glob(os.path.join(real_dir, "**/*.wav"), recursive=True) + \
                     glob.glob(os.path.join(real_dir, "**/*.mp3"), recursive=True) + \
                     glob.glob(os.path.join(real_dir, "**/*.flac"), recursive=True)
        
        cloned_files = glob.glob(os.path.join(cloned_dir, "**/*.wav"), recursive=True) + \
                       glob.glob(os.path.join(cloned_dir, "**/*.mp3"), recursive=True) + \
                       glob.glob(os.path.join(cloned_dir, "**/*.flac"), recursive=True)

        if max_samples_per_class:
            real_files = real_files[:max_samples_per_class]
            cloned_files = cloned_files[:max_samples_per_class]

        file_list = [(f, 0) for f in real_files] + [(f, 1) for f in cloned_files]
        np.random.seed(42)
        np.random.shuffle(file_list)

        tab_list, mel_list, lfcc_list, labels = [], [], [], []
        feature_names = []

        for fpath, label in file_list:
            try:
                y, _ = self.audio_loader.load_audio(fpath)
                tab_vec, feat_names = self.extractor.extract_tabular(y)
                mel_spec = self.extractor.extract_mel_spectrogram(y)
                lfcc_tensor = self.extractor.extract_lfcc_tensor(y)

                tab_list.append(tab_vec)
                mel_list.append(mel_spec)
                lfcc_list.append(lfcc_tensor)
                labels.append(label)
                if not feature_names:
                    feature_names = feat_names
            except Exception as e:
                print(f"Skipping {fpath} due to extraction error: {e}")

        X_tab = np.array(tab_list, dtype=np.float32)
        X_mel = np.array(mel_list, dtype=np.float32)
        X_lfcc = np.array(lfcc_list, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)

        return X_tab, X_mel, X_lfcc, y, feature_names

    def generate_synthetic_dataset(
        self,
        n_samples_per_class: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Generates in-memory procedural genuine vs synthetic cloned dataset for rapid training.
        """
        gen = SyntheticAudioGenerator(sr=self.sr)
        tab_list, mel_list, lfcc_list, labels = [], [], [], []
        feature_names = []

        # 1. Generate Genuine Samples
        for i in range(n_samples_per_class):
            base_f0 = np.random.uniform(90.0, 240.0)
            dur = np.random.uniform(2.5, 3.5)
            y_real = gen.generate_human_like_voice(duration=dur, base_f0=base_f0)

            tab_vec, feat_names = self.extractor.extract_tabular(y_real)
            mel_spec = self.extractor.extract_mel_spectrogram(y_real)
            lfcc_tensor = self.extractor.extract_lfcc_tensor(y_real)

            tab_list.append(tab_vec)
            mel_list.append(mel_spec)
            lfcc_list.append(lfcc_tensor)
            labels.append(0)
            if not feature_names:
                feature_names = feat_names

        # 2. Generate Cloned Samples with different artifact classes
        artifact_types = ["neural_vocoder", "griffin_lim", "voice_conversion"]
        for i in range(n_samples_per_class):
            base_f0 = np.random.uniform(90.0, 240.0)
            dur = np.random.uniform(2.5, 3.5)
            art = artifact_types[i % len(artifact_types)]

            base = gen.generate_human_like_voice(duration=dur, base_f0=base_f0)
            y_cloned = gen.generate_cloned_voice_artifacts(base, duration=dur, artifact_type=art)

            tab_vec, _ = self.extractor.extract_tabular(y_cloned)
            mel_spec = self.extractor.extract_mel_spectrogram(y_cloned)
            lfcc_tensor = self.extractor.extract_lfcc_tensor(y_cloned)

            tab_list.append(tab_vec)
            mel_list.append(mel_spec)
            lfcc_list.append(lfcc_tensor)
            labels.append(1)

        # Shuffle
        indices = np.arange(len(labels))
        np.random.seed(42)
        np.random.shuffle(indices)

        X_tab = np.array(tab_list, dtype=np.float32)[indices]
        X_mel = np.array(mel_list, dtype=np.float32)[indices]
        X_lfcc = np.array(lfcc_list, dtype=np.float32)[indices]
        y = np.array(labels, dtype=np.int64)[indices]

        return X_tab, X_mel, X_lfcc, y, feature_names
