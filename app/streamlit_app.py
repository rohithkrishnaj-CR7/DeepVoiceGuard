"""
DeepVoiceGuard: AI Voice Cloning & Audio Deepfake Detection Web Application.
Interactive Streamlit Dashboard with Real-Time Audio Scanning, Forensic Diagnostics,
Spectrogram Inspections, Batch File Audits, and Model Benchmark Insights.
"""

import os
import sys

# Ensure project root is always in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import json
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.models.ensemble import DeepVoiceGuard
from src.forensics.analyzer import ForensicAnalyzer
from src.features.audio_loader import AudioLoader, DEFAULT_SAMPLE_RATE

# Streamlit Page Config
st.set_page_config(
    page_title="DeepVoiceGuard | AI Voice Cloning Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic UI)
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .verdict-box-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .verdict-box-suspicious {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.25));
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    tab_path = os.path.join(PROJECT_ROOT, "saved_models", "tabular_model.joblib")
    lcnn_path = os.path.join(PROJECT_ROOT, "saved_models", "lcnn_model.pt")
    spec_path = os.path.join(PROJECT_ROOT, "saved_models", "specresnet_model.pt")

    detector = DeepVoiceGuard(
        tabular_model_path=tab_path if os.path.exists(tab_path) else None,
        lcnn_model_path=lcnn_path if os.path.exists(lcnn_path) else None,
        specresnet_model_path=spec_path if os.path.exists(spec_path) else None
    )
    return detector

detector = load_detector()
forensics_engine = ForensicAnalyzer()
audio_loader = AudioLoader()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=70)
    st.markdown("### DeepVoiceGuard AI")
    st.caption("Voice Anti-Spoofing & Deepfake Audio Forensics")
    st.divider()

    st.markdown("#### ⚡ Active Model Status")
    if detector.tabular_model and detector.tabular_model.is_fitted:
        st.success("✔ Tabular Ensemble (LGBM+XGB+RF)", icon="✅")
    else:
        st.warning("⚠ Tabular Model (Untrained)", icon="⚠️")

    if detector.lcnn_model and detector.lcnn_model.is_fitted:
        st.success("✔ PyTorch LFCC-LCNN DeepNet", icon="✅")
    else:
        st.warning("⚠ LFCC-LCNN (Untrained)", icon="⚠️")

    if detector.specresnet_model and detector.specresnet_model.is_fitted:
        st.success("✔ SpecResNet Attention Net", icon="✅")
    else:
        st.warning("⚠ SpecResNet (Untrained)", icon="⚠️")

    st.divider()
    st.markdown("#### ⚙️ Scanner Sensitivity")
    chunk_dur = st.slider("Segment Window (seconds)", min_value=1.5, max_value=5.0, value=3.0, step=0.5)
    overlap_rate = st.slider("Window Overlap", min_value=0.0, max_value=0.75, value=0.50, step=0.25)
    st.divider()
    st.caption("v1.0.0 | Academic & Production Ready")

# Main Header
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="main-title">🛡️ DeepVoiceGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced AI Voice Cloning & Synthetic Speech Detection System</div>', unsafe_allow_html=True)

# Tabs
tab_scan, tab_forensics, tab_demo, tab_batch, tab_benchmarks = st.tabs([
    "🎙️ Live Audio Scanner",
    "🔬 Forensic Deep-Dive",
    "🧪 Demo Sample Lab",
    "📁 Batch File Scanner",
    "📊 Benchmarks & Architecture"
])

# ----------------- TAB 1: LIVE SCANNER -----------------
with tab_scan:
    st.markdown("#### Upload Audio or Record Voice for Instant Deepfake Detection")
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload audio file (.wav, .mp3, .flac, .ogg, .m4a)",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            key="scanner_upload"
        )

    with col_input2:
        mic_audio = st.audio_input("Or Record via Microphone")

    active_audio = uploaded_file or mic_audio

    # Check if a sample was triggered from Demo tab
    if active_audio is None and 'auto_scan_bytes' in st.session_state:
        active_audio = io.BytesIO(st.session_state['auto_scan_bytes'])

    if active_audio is not None:
        audio_bytes = active_audio.read() if hasattr(active_audio, 'read') else active_audio.getvalue()
        st.audio(audio_bytes)

        with st.spinner("Analyzing acoustic features, LFCCs, harmonic stability, and vocoder artifacts..."):
            try:
                scan_res = detector.scan_audio(io.BytesIO(audio_bytes), chunk_duration=chunk_dur, overlap=overlap_rate)
                radar_scores = forensics_engine.compute_forensic_radar(scan_res['forensics'])
                findings = forensics_engine.generate_forensic_summary(scan_res)

                st.session_state['latest_scan'] = scan_res
                st.session_state['latest_audio'] = scan_res['raw_audio']
                st.session_state['latest_radar'] = radar_scores
                st.session_state['latest_findings'] = findings

                prob_fake = scan_res['cloned_probability']
                prob_real = scan_res['real_probability']
                confidence = scan_res['confidence_score']
                risk = scan_res['risk_level']
                verdict = scan_res['verdict']

                st.markdown("---")
                if verdict == "AI_CLONED_SYNTHETIC":
                    st.markdown(f"""
                    <div class="verdict-box-fake">
                        <h2 style="color: #ef4444; margin:0;">🚨 AI CLONED / SYNTHETIC VOICE DETECTED</h2>
                        <p style="font-size: 1.15rem; margin-top: 5px; color: #fca5a5;">
                            High probability of neural speech synthesis, voice conversion, or deepfake vocoder manipulation.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "GENUINE_HUMAN_VOICE":
                    st.markdown(f"""
                    <div class="verdict-box-real">
                        <h2 style="color: #10b981; margin:0;">✅ AUTHENTIC HUMAN VOICE CONFIRMED</h2>
                        <p style="font-size: 1.15rem; margin-top: 5px; color: #6ee7b7;">
                            Acoustic signals, pitch micro-dynamics, and high-frequency harmonics conform to natural vocal biology.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-box-suspicious">
                        <h2 style="color: #f59e0b; margin:0;">⚠️ SUSPICIOUS ACOUSTIC ANOMALIES</h2>
                        <p style="font-size: 1.15rem; margin-top: 5px; color: #fde68a;">
                            Inconclusive characteristics or partial synthetic cues detected. Further inspection recommended.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Cloned Probability", f"{prob_fake * 100:.1f}%")
                with m2:
                    st.metric("Authentic Probability", f"{prob_real * 100:.1f}%")
                with m3:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                with m4:
                    risk_color = "red" if risk in ("HIGH", "CRITICAL") else ("orange" if risk == "MEDIUM" else "green")
                    st.metric("Threat Level", risk)

                # Gauge & Segment Timeline
                col_g, col_t = st.columns([1, 1.5])
                with col_g:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_fake * 100.0,
                        title={'text': "Cloned Probability Score (%)", 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444" if prob_fake >= 0.5 else "#10b981"},
                            'steps': [
                                {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.25)"},
                                {'range': [35, 65], 'color': "rgba(245, 158, 11, 0.25)"},
                                {'range': [65, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                            ],
                            'threshold': {
                                'line': {'color': "white", 'width': 3},
                                'thickness': 0.75,
                                'value': 50.0
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with col_t:
                    st.markdown("##### ⏱️ Temporal Segment Anomaly Heatmap")
                    timeline = scan_res['segment_timeline']
                    df_time = pd.DataFrame(timeline)
                    if not df_time.empty:
                        df_time['Segment'] = [f"{r['start_time']}s - {r['end_time']}s" for _, r in df_time.iterrows()]
                        df_time['Risk %'] = df_time['cloned_probability'] * 100.0
                        
                        fig_time = px.bar(
                            df_time,
                            x='Segment',
                            y='Risk %',
                            color='Risk %',
                            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                            range_color=[0, 100],
                            labels={'Risk %': 'Synthetic Probability %'}
                        )
                        fig_time.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Threshold (50%)")
                        fig_time.update_layout(height=250, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_time, use_container_width=True)

                # Forensic Bullet Findings
                st.markdown("##### 🔎 Key Forensic Diagnostic Findings")
                for item in findings:
                    st.markdown(f"- {item}")

            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
    else:
        st.info("💡 Upload an audio file or record from microphone above, or switch to the **Demo Sample Lab** tab to test preloaded samples!")

# ----------------- TAB 2: FORENSIC DEEP-DIVE -----------------
with tab_forensics:
    st.markdown("#### 🔬 Detailed Acoustic Forensic & Spectral Inspection")

    if 'latest_scan' in st.session_state:
        scan_data = st.session_state['latest_scan']
        raw_y = st.session_state['latest_audio']
        radar_data = st.session_state['latest_radar']

        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            st.markdown("##### 🕸️ Forensic Acoustic Radar Profile")
            categories = list(radar_data.keys())
            values = list(radar_data.values())
            categories.append(categories[0])
            values.append(values[0])

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Current Audio',
                line_color='#00f2fe',
                fillcolor='rgba(0, 242, 254, 0.25)'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[85, 85, 85, 85, 85, 85, 85],
                theta=categories,
                name='Natural Human Baseline',
                line=dict(color='#10b981', dash='dash')
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=350,
                margin=dict(l=40, r=40, t=30, b=30),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_f2:
            st.markdown("##### 🎵 Fundamental Frequency (F0) & Pitch Jitter Track")
            p_data = forensics_engine.compute_pitch_and_hnr_track(raw_y)
            df_pitch = pd.DataFrame({'Time (s)': p_data['times'], 'F0 (Hz)': p_data['f0']})
            df_pitch = df_pitch[df_pitch['F0 (Hz)'] > 0]

            if not df_pitch.empty:
                fig_pitch = px.line(df_pitch, x='Time (s)', y='F0 (Hz)', title="Voiced Pitch Trajectory")
                fig_pitch.add_hrect(y0=85, y1=255, fillcolor="rgba(16, 185, 129, 0.1)", line_width=0, annotation_text="Typical Human Vocal Band (85-255Hz)")
                fig_pitch.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pitch, use_container_width=True)
            else:
                st.info("No periodic voiced frames detected for pitch tracking.")

        # Interactive Spectrogram Visualizer
        st.markdown("##### 🌈 Time-Frequency Spectrogram Analysis")
        spec_data = forensics_engine.compute_spectrograms(raw_y)
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("###### Log-Mel Spectrogram (Acoustic Resonance)")
            fig_mel = px.imshow(
                spec_data['mel_spectrogram'],
                origin='lower',
                aspect='auto',
                color_continuous_scale='Magma',
                labels={'x': 'Time Frames', 'y': 'Mel Frequency Bins'}
            )
            fig_mel.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_mel, use_container_width=True)

        with col_s2:
            st.markdown("###### Linear STFT Spectrogram (High-Frequency Cutoff Inspector)")
            fig_lin = px.imshow(
                spec_data['linear_spectrogram'],
                origin='lower',
                aspect='auto',
                color_continuous_scale='Viridis',
                labels={'x': 'Time Frames', 'y': 'Linear Frequency Bins'}
            )
            fig_lin.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_lin, use_container_width=True)

        # Tabular Descriptors Table
        st.markdown("##### 📋 Raw Acoustic & Forensic Descriptors")
        forensic_table = pd.DataFrame([
            {"Descriptor": k, "Measured Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in scan_data['forensics'].items()
            if not k.startswith("mfcc_") and not k.startswith("lfcc_")
        ])
        st.dataframe(forensic_table, use_container_width=True, height=250)

    else:
        st.info("Scan an audio file in Tab 1 or Tab 3 to view deep forensic diagnostics here.")

# ----------------- TAB 3: DEMO SAMPLE LAB -----------------
with tab_demo:
    st.markdown("#### 🧪 Ready-to-Test Audio Sample Gallery")
    st.caption("Instantly test verified Genuine Human recordings against AI-Cloned voice samples without uploading files.")

    sample_real_dir = os.path.join(PROJECT_ROOT, "sample_data", "real")
    sample_cloned_dir = os.path.join(PROJECT_ROOT, "sample_data", "cloned")

    col_dr, col_df = st.columns(2)

    with col_dr:
        st.markdown("##### 🟢 Genuine Human Voice Samples")
        real_files = glob.glob(os.path.join(sample_real_dir, "*.wav")) if os.path.exists(sample_real_dir) else []
        if real_files:
            for rf in real_files:
                bname = os.path.basename(rf)
                st.markdown(f"**{bname}**")
                st.audio(rf)
                if st.button(f"🔍 Scan Sample: {bname}", key=f"btn_{bname}"):
                    with open(rf, 'rb') as f:
                        st.session_state['auto_scan_bytes'] = f.read()
                        st.success(f"Loaded {bname}! Switch to Tab 1 to view results.")
        else:
            st.warning("No sample files found. Run 'python run_demo.py' to generate demo files.")

    with col_df:
        st.markdown("##### 🔴 AI-Cloned & Synthetic Voice Samples")
        cloned_files = glob.glob(os.path.join(sample_cloned_dir, "*.wav")) if os.path.exists(sample_cloned_dir) else []
        if cloned_files:
            for cf in cloned_files:
                bname = os.path.basename(cf)
                st.markdown(f"**{bname}**")
                st.audio(cf)
                if st.button(f"🔍 Scan Sample: {bname}", key=f"btn_{bname}"):
                    with open(cf, 'rb') as f:
                        st.session_state['auto_scan_bytes'] = f.read()
                        st.success(f"Loaded {bname}! Switch to Tab 1 to view results.")
        else:
            st.warning("No sample files found. Run 'python run_demo.py' to generate demo files.")

# ----------------- TAB 4: BATCH FILE SCANNER -----------------
with tab_batch:
    st.markdown("#### 📁 High-Throughput Batch Audio Scanner")
    st.caption("Upload multiple audio recordings (e.g. KYC voice verification, Call center forensics) to generate an automated audit report.")

    batch_files = st.file_uploader(
        "Upload multiple audio files",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        if st.button(f"🚀 Run Batch Scan on {len(batch_files)} Files"):
            prog_bar = st.progress(0)
            status_txt = st.empty()

            batch_results = []
            for idx, b_file in enumerate(batch_files):
                status_txt.text(f"Scanning ({idx+1}/{len(batch_files)}): {b_file.name}")
                try:
                    b_bytes = b_file.read()
                    res = detector.scan_audio(io.BytesIO(b_bytes))
                    batch_results.append({
                        'Filename': b_file.name,
                        'Verdict': res['verdict'],
                        'Cloned Probability (%)': round(res['cloned_probability'] * 100, 2),
                        'Confidence (%)': round(res['confidence_score'], 1),
                        'Risk Level': res['risk_level'],
                        'Duration (s)': res['audio_duration']
                    })
                except Exception as e:
                    batch_results.append({
                        'Filename': b_file.name,
                        'Verdict': f"ERROR: {e}",
                        'Cloned Probability (%)': 0.0,
                        'Confidence (%)': 0.0,
                        'Risk Level': 'UNKNOWN',
                        'Duration (s)': 0.0
                    })
                prog_bar.progress((idx + 1) / len(batch_files))

            status_txt.text("Batch audit completed!")
            df_batch = pd.DataFrame(batch_results)

            st.markdown("##### 📊 Batch Audit Summary")
            c_tot, c_real, c_fake = st.columns(3)
            with c_tot:
                st.metric("Total Files", len(df_batch))
            with c_real:
                n_real = len(df_batch[df_batch['Verdict'] == 'GENUINE_HUMAN_VOICE'])
                st.metric("Genuine Human", n_real)
            with c_fake:
                n_fake = len(df_batch[df_batch['Verdict'] == 'AI_CLONED_SYNTHETIC'])
                st.metric("AI Cloned / Fake", n_fake)

            st.dataframe(df_batch, use_container_width=True)

            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV Audit Report",
                data=csv_data,
                file_name="deepvoiceguard_batch_report.csv",
                mime="text/csv"
            )

# ----------------- TAB 5: BENCHMARKS & ARCHITECTURE -----------------
with tab_benchmarks:
    st.markdown("#### 📊 Model Performance Benchmarks & Forensic Architecture")

    metrics_file = os.path.join(PROJECT_ROOT, "saved_models", "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)

        ens_m = metrics_data.get('ensemble', {})
        st.markdown("##### 🏆 Test Set Performance Metrics")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.metric("Equal Error Rate (EER)", f"{ens_m.get('eer_percentage', 0.0)}%")
        with b2:
            st.metric("ROC-AUC Score", f"{ens_m.get('roc_auc', 0.0):.4f}")
        with b3:
            st.metric("Accuracy", f"{ens_m.get('accuracy', 0.0)*100:.2f}%")
        with b4:
            st.metric("F1-Score", f"{ens_m.get('f1_score', 0.0):.4f}")

        # ROC Curve Plot
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("##### 📈 Receiver Operating Characteristic (ROC) Curve")
            roc = ens_m.get('roc_curve', {})
            if roc and 'fpr' in roc:
                df_roc = pd.DataFrame({'False Positive Rate': roc['fpr'], 'True Positive Rate': roc['tpr']})
                fig_roc = px.area(df_roc, x='False Positive Rate', y='True Positive Rate', title=f"Ensemble ROC Curve (AUC = {ens_m.get('roc_auc', 0.0):.4f})")
                fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_roc, use_container_width=True)

        with col_r2:
            st.markdown("##### 🌟 Top Predictive Acoustic Features")
            feats = metrics_data.get('feature_importances', [])[:12]
            if feats:
                df_feat = pd.DataFrame(feats, columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig_feat = px.bar(df_feat, x='Importance', y='Feature', orientation='h', title="Feature Importance Breakdown")
                fig_feat.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_feat, use_container_width=True)

    else:
        st.info("Run 'python run_demo.py' to calibrate models and generate benchmark plots.")

    st.divider()
    st.markdown("""
    #### 🏗️ Technical Architecture & Signal Processing Pipeline
    
    1. **Multi-Domain Feature Extraction**:
       - **Linear Frequency Cepstral Coefficients (LFCC)**: Extracts 60-dimensional spectral coefficients linearly spaced to capture high-frequency vocoder phase distortion.
       - **Mel-Frequency Cepstral Coefficients (MFCC)**: 20 MFCCs + 10 Deltas for spectral envelope analysis.
       - **Pitch (F0) & Harmonics**: Tracks fundamental vocal frequency, pitch micro-jitter, and Harmonic-to-Noise Ratio (HNR).
       - **Sub-Band Energy Ratios**: Identifies sharp high-frequency vocoder cutoffs (<7kHz) and unnatural diffusion noise floors.
    
    2. **Multi-Model Hybrid Classifier Suite**:
       - **Tabular GBDT Ensemble**: LightGBM + XGBoost + Random Forest + Extra Trees trained on 161 extracted acoustic features.
       - **LFCC-LCNN Deep Network**: PyTorch Convolutional Neural Network with Max-Feature-Map (MFM) activation units specifically designed for speech anti-spoofing.
       - **SpecResNet**: 2D Spectrogram Residual Network for deep spectral pattern recognition.
    
    3. **Confidence Calibration & Forensic Explainability**:
       - Segment-level sliding-window anomaly localization.
       - Physics-grounded radar metric scoring (Pitch Naturalness, High-Freq Integrity, Harmonic Richness, etc.).
    """)
