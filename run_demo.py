"""
DeepVoiceGuard Demo Setup & Quickstart Script.
Generates audio samples, trains baseline models, and performs a test verification scan.
"""

import os
import sys

from src.data.synthetic_generator import SyntheticAudioGenerator
from src.data.dataset_loader import VoiceDatasetLoader
from src.training.trainer import ModelTrainer
from src.models.ensemble import DeepVoiceGuard

def main():
    print("=" * 60)
    print("  DeepVoiceGuard: Quickstart Setup & Calibration")
    print("=" * 60)

    # 1. Generate Demo Audio Files
    print("\n[Step 1/3] Generating genuine and cloned demo audio samples in sample_data/...")
    gen = SyntheticAudioGenerator()
    demo_files = gen.generate_demo_samples()

    # 2. Synthesize calibration dataset & Train models
    print("\n[Step 2/3] Generating calibration dataset and training models...")
    loader = VoiceDatasetLoader()
    X_tab, X_mel, X_lfcc, y, feat_names = loader.generate_synthetic_dataset(n_samples_per_class=40)

    trainer = ModelTrainer(output_dir="saved_models")
    metrics = trainer.train_all_models(X_tab, X_mel, X_lfcc, y, feature_names=feat_names, epochs=10)

    # 3. Test verification scan
    print("\n[Step 3/3] Running verification test scan on sample cloned audio...")
    detector = DeepVoiceGuard(
        tabular_model_path="saved_models/tabular_model.joblib",
        lcnn_model_path="saved_models/lcnn_model.pt",
        specresnet_model_path="saved_models/specresnet_model.pt"
    )

    test_file = demo_files['cloned'][0]
    res = detector.scan_audio(test_file)

    print("\n" + "-" * 50)
    print(f"  Test Audio File:      {os.path.basename(test_file)}")
    print(f"  Verdict:              {res['verdict']}")
    print(f"  Cloned Probability:   {res['cloned_probability'] * 100:.2f}%")
    print(f"  Confidence Score:     {res['confidence_score']:.1f}%")
    print(f"  Risk Level:           {res['risk_level']}")
    print("-" * 50)

    print("\n[SUCCESS] Setup complete! You can now launch the web dashboard:")
    print("   streamlit run app/streamlit_app.py\n")

if __name__ == "__main__":
    main()
