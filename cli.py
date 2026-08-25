"""
DeepVoiceGuard Command-Line Interface (CLI).
Scan audio files, run batch audits, train models, and manage datasets from the terminal.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import json
import glob
import pandas as pd
from typing import Optional

from src.models.ensemble import DeepVoiceGuard
from src.forensics.analyzer import ForensicAnalyzer
from src.data.synthetic_generator import SyntheticAudioGenerator
from src.data.dataset_loader import VoiceDatasetLoader
from src.training.trainer import ModelTrainer

def print_banner():
    print(r"""
========================================================================
   ____                 _   _       _            ____                     _ 
  |  _ \  ___  ___ _ __| | | | ___ (_) ___ ___  / ___|_   _  __ _ _ __ __| |
  | | | |/ _ \/ _ \ '_ \ | | |/ _ \| |/ __/ _ \| |  _| | | |/ _` | '__/ _` |
  | |_| |  __/  __/ |_) \ \_/ / (_) | | (_|  __/| |_| | |_| | (_| | | | (_| |
  |____/ \___|\___| .__/ \___/ \___/|_|\___\___(_)____|\__,_|\__,_|_|  \__,_|
                  |_|        AI Voice Cloning & Deepfake Detection System
========================================================================
    """)

def get_detector() -> DeepVoiceGuard:
    tab_path = "saved_models/tabular_model.joblib"
    lcnn_path = "saved_models/lcnn_model.pt"
    spec_path = "saved_models/specresnet_model.pt"

    return DeepVoiceGuard(
        tabular_model_path=tab_path if os.path.exists(tab_path) else None,
        lcnn_model_path=lcnn_path if os.path.exists(lcnn_path) else None,
        specresnet_model_path=spec_path if os.path.exists(spec_path) else None
    )

def cmd_scan(args):
    print_banner()
    if not os.path.exists(args.file):
        print(f"Error: File not found -> {args.file}")
        sys.exit(1)

    print(f"[*] Scanning audio file: {args.file}")
    detector = get_detector()
    forensic = ForensicAnalyzer()

    res = detector.scan_audio(args.file)
    radar = forensic.compute_forensic_radar(res['forensics'])
    findings = forensic.generate_forensic_summary(res)

    print("\n" + "=" * 55)
    print(f"  DETECTION VERDICT:     {res['verdict']}")
    print(f"  CLONED PROBABILITY:    {res['cloned_probability'] * 100.0:.2f}%")
    print(f"  CONFIDENCE SCORE:      {res['confidence_score']:.1f}%")
    print(f"  RISK ASSESSMENT:       {res['risk_level']}")
    print(f"  AUDIO DURATION:        {res['audio_duration']} seconds")
    print(f"  SEGMENTS ANALYZED:     {res['num_segments']}")
    print("=" * 55)

    print("\n[+] Acoustic Forensic Profile (0-100 Score):")
    for k, v in radar.items():
        filled = int(v / 5)
        bar = "#" * filled + "-" * (20 - filled)
        print(f"  - {k:<22} [{bar}] {v:.1f}")

    print("\n[+] Forensic Findings:")
    for b in findings:
        clean_b = b.replace("**", "")
        print(f"  * {clean_b}")

    if args.json:
        out_path = args.json
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'file': args.file,
                'verdict': res['verdict'],
                'cloned_prob': res['cloned_probability'],
                'confidence': res['confidence_score'],
                'risk_level': res['risk_level'],
                'radar': radar,
                'findings': findings,
                'timeline': res['segment_timeline']
            }, f, indent=2)
        print(f"\n[v] Saved JSON report to: {out_path}")

def cmd_batch(args):
    print_banner()
    if not os.path.isdir(args.dir):
        print(f"Error: Directory not found -> {args.dir}")
        sys.exit(1)

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
        audio_files.extend(glob.glob(os.path.join(args.dir, ext)))

    if not audio_files:
        print(f"No audio files (.wav, .mp3, .flac, .ogg) found in {args.dir}")
        return

    print(f"[*] Processing batch of {len(audio_files)} audio files in {args.dir}...")
    detector = get_detector()

    records = []
    for fpath in audio_files:
        try:
            res = detector.scan_audio(fpath)
            records.append({
                'filename': os.path.basename(fpath),
                'filepath': fpath,
                'verdict': res['verdict'],
                'is_cloned': res['is_cloned'],
                'cloned_probability': res['cloned_probability'],
                'confidence_score': res['confidence_score'],
                'risk_level': res['risk_level'],
                'duration_sec': res['audio_duration']
            })
            print(f"  [+] {os.path.basename(fpath):<35} -> {res['verdict']:<22} ({res['cloned_probability']*100:.1f}%)")
        except Exception as e:
            print(f"  [-] Error scanning {fpath}: {e}")

    df = pd.DataFrame(records)
    out_csv = args.out or "batch_scan_report.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[v] Batch audit complete! Exported report for {len(records)} files to: {out_csv}")

def cmd_demo_setup(args):
    print_banner()
    print("[*] Generating benchmark dataset and sample audio files...")
    gen = SyntheticAudioGenerator()
    gen.generate_demo_samples()

    print("\n[*] Synthesizing calibration training dataset...")
    loader = VoiceDatasetLoader()
    X_tab, X_mel, X_lfcc, y, feat_names = loader.generate_synthetic_dataset(n_samples_per_class=args.samples)

    print("\n[*] Training DeepVoiceGuard Model Suite (Tabular + LCNN + SpecResNet)...")
    trainer = ModelTrainer(output_dir="saved_models")
    trainer.train_all_models(X_tab, X_mel, X_lfcc, y, feature_names=feat_names, epochs=args.epochs)
    print("[v] Setup complete! Models and sample data are ready.")

def main():
    parser = argparse.ArgumentParser(description="DeepVoiceGuard Voice Cloning Detection CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Scan command
    scan_p = subparsers.add_parser("scan", help="Scan a single audio file")
    scan_p.add_argument("--file", "-f", required=True, help="Path to audio file (.wav, .mp3, .flac)")
    scan_p.add_argument("--json", "-j", default=None, help="Optional output JSON path")

    # Batch command
    batch_p = subparsers.add_parser("scan-batch", help="Scan all audio files in a directory")
    batch_p.add_argument("--dir", "-d", required=True, help="Directory containing audio files")
    batch_p.add_argument("--out", "-o", default="batch_report.csv", help="Output CSV report path")

    # Demo setup / Train command
    setup_p = subparsers.add_parser("demo-setup", help="Generate sample clips and train baseline models")
    setup_p.add_argument("--samples", type=int, default=50, help="Number of training samples per class")
    setup_p.add_argument("--epochs", type=int, default=12, help="Number of deep model training epochs")

    args = parser.parse_args()
    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "scan-batch":
        cmd_batch(args)
    elif args.command == "demo-setup":
        cmd_demo_setup(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
