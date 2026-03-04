import streamlit as st
import openai
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sounddevice as sd
import soundfile as sf
import hashlib
import time
import os
import io
from scipy.io import wavfile
from cryptography.hazmat.primitives import serialization
from audio_warp.config import WatermarkConfig
from audio_warp.spread_spectrum import (
    generate_pn_sequence, 
    embed_bits_in_frame, 
    extract_bits_from_frame
)
from audio_warp.audio_io import to_canonical
from audio_warp.crypto import generate_keypair, sign, verify, audio_hash
from audio_warp.attacks import apply_attack, ATTACK_REGISTRY
from audio_warp.embedder import embed
from audio_warp.detector import detect
from audio_warp.payload import build_payload
from audio_warp.ecc import encode, decode

# --- Global Page Config ---
st.set_page_config(page_title="RIMH work 🧪", layout="wide", page_icon="🧪")

# --- Custom Styling (Premium UI) ---
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-image: radial-gradient(circle at 2% 10%, rgba(31, 119, 180, 0.05), transparent 25%),
                          radial-gradient(circle at 98% 90%, rgba(255, 127, 14, 0.05), transparent 25%);
    }
    
    /* Card/Container Styling */
    /* Premium Tab Navigation Redesign */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(26, 28, 35, 0.8) !important;
        backdrop-filter: blur(10px);
        padding: 8px 16px !important;
        border-radius: 50px !important;
        border: 1px solid rgba(255, 255, 255, 0.05);
        display: inline-flex !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        gap: 12px !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px !important;
        border-radius: 40px !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        background: transparent !important;
        padding: 0 24px !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05rem !important;
        text-transform: uppercase !important;
        color: #808495 !important;
        transition: all 0.3s ease !important;
        margin: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(31, 119, 180, 0.15) !important;
        box-shadow: inset 0 0 10px rgba(31, 119, 180, 0.2) !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
        text-shadow: 0 0 15px rgba(31, 119, 180, 0.5) !important;
    }
    .stTabs [data-baseweb="tab"]:hover p {
        color: #ffffff !important;
        letter-spacing: 0.1rem !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 500;
        color: #f0f2f6;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #808495 !important;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(26, 28, 35, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    /* Hero Headers */
    .hero-container {
        padding: 2rem 1.5rem;
        background: linear-gradient(90deg, rgba(31, 119, 180, 0.1) 0%, rgba(38, 39, 48, 0.4) 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #1f77b4;
    }
    .hero-title {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #a0a6b5;
        font-size: 1.1rem;
    }

    /* Lab Cards (Replacement for Expanders) */
    .lab-card {
        background: rgba(38, 39, 48, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-left: 4px solid #1f77b4;
    }
    .lab-card:hover {
        transform: translateY(-2px);
        background: rgba(45, 47, 58, 0.6);
        border-color: rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    .card-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 15px;
    }
    .card-icon {
        font-size: 1.5rem;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .card-accent-blue { border-left-color: #3b82f6; }
    .card-accent-green { border-left-color: #10b981; }
    .card-accent-purple { border-left-color: #a855f7; }
    .card-accent-amber { border-left-color: #f59e0b; }
    .card-accent-ruby { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# --- Directory Setup ---
for d in ["original", "watermarked", "attacked", "logs", "keys"]:
    if not os.path.exists(d):
        os.makedirs(d)

# --- Session State Initialization ---
def init_session():
    if 'config' not in st.session_state:
        st.session_state.config = WatermarkConfig()
        st.session_state.config.chips_per_bit = 63
        st.session_state.config.embed_strength = 0.2
        
    if 'owner_id' not in st.session_state:
        st.session_state.owner_id = "STUDENT1"
    
    if 'audio' not in st.session_state:
        st.session_state.audio = None
    if 'sr' not in st.session_state:
        st.session_state.sr = 44100
    if 'wm_audio' not in st.session_state:
        st.session_state.wm_audio = None
    if 'attacked_audio' not in st.session_state:
        st.session_state.attacked_audio = None
    
    if 'users' not in st.session_state:
        priv, pub = generate_keypair()
        st.session_state.users = {
            "Owner": {"priv": priv, "pub": pub, "name": "Ishu (Authorized)"},
            "Adversary": {"priv": generate_keypair()[0], "pub": generate_keypair()[1], "name": "Hima (Un-Authorized)"}
        }
    if 'active_user' not in st.session_state:
        st.session_state.active_user = "Owner"
    
    if 'openai_key' not in st.session_state:
        # Load from secrets if available, otherwise leave empty for user input
        st.session_state.openai_key = st.secrets.get("OPENAI_API_KEY", "")
    
    if 'ai_explanation' not in st.session_state:
        st.session_state.ai_explanation = ""
    
    if 'last_metrics' not in st.session_state:
        st.session_state.last_metrics = {}

    # Localized Forensic State (Act 1, Studio, Detection, Attack)
    for act in ["act1", "studio", "detect", "atk"]:
        if f'metrics_{act}' not in st.session_state: st.session_state[f'metrics_{act}'] = {}
        if f'res_{act}' not in st.session_state: st.session_state[f'res_{act}'] = None
        if f'recovery_msg_{act}' not in st.session_state: st.session_state[f'recovery_msg_{act}'] = None
        if f'recovery_type_{act}' not in st.session_state: st.session_state[f'recovery_type_{act}'] = None

init_session()

# --- Utility Functions ---
def get_audio_path(audio, sr, name, folder="original"):
    path = os.path.join(folder, f"{name}.wav")
    sf.write(path, audio, sr, format='wav')
    return path

def calculate_snr(orig, mod):
    if orig is None or mod is None: return 0.0
    min_len = min(len(orig), len(mod))
    orig_p = np.sum(orig[:min_len]**2)
    diff_p = np.sum((orig[:min_len] - mod[:min_len])**2)
    if diff_p == 0: return 100.0
    return 10 * np.log10(orig_p / (diff_p + 1e-10))

def ensure_8_bytes(id_str: str) -> bytes:
    """Ensure string is exactly 8 bytes by padding or truncating."""
    b = id_str.encode()
    if len(b) > 8:
        return b[:8]
    return b.ljust(8, b"_")

def openai_explanation(metrics, context="Embedding"):
    """Fetch pedagogical explanation from OpenAI GPT-4."""
    if not st.session_state.openai_key:
        return "⚠️ OpenAI API Key missing. Please provide it in the sidebar."
    
    try:
        client = OpenAI(api_key=st.session_state.openai_key)
        
        prompt = f"""
        You are an expert in Digital Audio Watermarking and Information Hiding. 
        Explain the following experimental results to a university student in a pedagogical, insightful way.
        
        Context: {context}
        Metrics:
        - SNR (Signal-to-Noise Ratio): {metrics.get('snr', 'N/A')}
        - Mean Confidence Score: {metrics.get('conf', 'N/A')}
        - Bits Embedded: {metrics.get('bits', 'N/A')}
        - Embedding Strength (Alpha): {st.session_state.config.embed_strength}
        - Redundancy (Chips/Bit): {st.session_state.config.chips_per_bit}
        - Signature Status: {metrics.get('sig', 'N/A')}
        - ECC Corrections: {metrics.get('ecc', 'N/A')}
        
        Explain:
        1. Whether these results are "good" or "bad".
        2. The tradeoff between quality (SNR) and detectability (Confidence).
        3. Simple advice on how to improve the results (e.g., increase alpha, decrease chips).
        Limit to 4-5 concise, professional bullet points.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful teaching assistant for an Audio Engineering lab."},
                      {"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg:
            return "❌ **AI Quota Exceeded**: Your OpenAI API key has run out of credits or reached its usage limit. Please check your billing status at [platform.openai.com](https://platform.openai.com/account/billing)."
        return f"❌ AI Error: {error_msg}"

# ===========================================================================
# GLOBAL COMPONENTS
# ===========================================================================
def standard_metric_panel(metrics, panel_id="main"):
    st.markdown("""<div style="margin-top: 2rem; margin-bottom: 1rem;"></div>""", unsafe_allow_html=True)
    with st.expander("📊 View Detailed Forensic Metrics", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Owner ID", st.session_state.owner_id)
        c1.metric("Embed Strength (α)", f"{st.session_state.config.embed_strength}")
        
        c2.metric("Chips per Bit", st.session_state.config.chips_per_bit)
        c2.metric("Bits Embedded", metrics.get('bits', '0'))
        
        c3.metric("Mean Confidence", f"{metrics.get('conf', 0):.4f}")
        snr = metrics.get('snr', 'N/A')
        if isinstance(snr, float): snr = f"{snr:.1f} dB"
        c3.metric("SNR (dB)", snr)
        
        c4.metric("ECC Corrections", metrics.get('ecc', '0'))
        sig = metrics.get('sig')
        sig_text = "✅ VALID" if sig is True else ("❌ INVALID" if sig is False else "N/A")
        c4.metric("Signature Status", sig_text)

        # AI EXPLANATION BUTTON
        exp_key = f"ai_exp_{panel_id}"
        if exp_key not in st.session_state:
            st.session_state[exp_key] = ""

        if st.button("🤖 AI Analysis: Explain these Results", key=f"ai_btn_{panel_id}", use_container_width=True):
            with st.spinner("Consulting AI Therapist..."):
                explanation = openai_explanation(metrics, context=f"Forensic Analysis: {panel_id}")
                st.session_state[exp_key] = explanation
        
        if st.session_state[exp_key]:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(st.session_state[exp_key])
                if st.button("Clear AI Analysis", key=f"clear_ai_{panel_id}"):
                    st.session_state[exp_key] = ""
                    st.rerun()

def global_sidebar():
    with st.sidebar:
        st.title("🛡️ Lab Control")
        st.markdown("---")
        
        st.subheader("⚙️ Global Parameters")
        st.session_state.config.embed_strength = st.slider(
            "Global Alpha (alpha)", 
            0.0, 1.0, 
            float(st.session_state.config.embed_strength), 
            help="Higher = more robust, but more audible distortion."
        )
        st.session_state.config.chips_per_bit = st.select_slider(
            "Global Chips per Bit", 
            options=[31, 63, 127], 
            value=st.session_state.config.chips_per_bit, 
            key="global_chips_sidebar_final"
        )
        
        toy_mode = st.checkbox("🎯 Toy Mode (Visual Clarity)", help="Forces high visibility parameters for demonstration.")
        if toy_mode:
            st.session_state.config.embed_strength = 0.8
            st.session_state.config.chips_per_bit = 31
            st.warning("🚀 Toy Mode: Max Visibility")


        st.markdown("---")
        st.subheader("📚 Watermark Cheat Sheet")
        st.markdown(r"""
| Term | Meaning | Role |
| :--- | :--- | :--- |
| **$\alpha$** | Strength | Robustness vs Quality |
| **PN** | Secret Key | Seeded +/- 1 pattern |
| **$C$** | Correlation | Alignment score |
| **SNR** | Quality | Host vs Watermark Ratio |
| **ECC** | Recovery | RS bit-flip correction |
        """)
        
        st.markdown("---")
        st.info("💡 Changes here affect all Acts in real-time.")

def standard_visualizations(audio, sr, title="Audio Forensics", wm_audio=None, res=None):
    st.markdown(f"#### {title}")
    v1, v2 = st.columns(2)
    
    with v1:
        if audio is not None:
            fig1, ax1 = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
            ax1.set_facecolor('#1a1c23')
            ax1.plot(audio[:int(sr*0.5)], color="#3b82f6", alpha=0.8)
            ax1.set_title("Time Domain Analysis (0.5s)", color='white', fontsize=10)
            ax1.grid(True, alpha=0.1, color='white')
            ax1.tick_params(colors='white', labelsize=8)
            st.pyplot(fig1)
        else:
            st.info("🕒 Waveform preview unavailable.")
        
        # Correlation Plot
        if res and hasattr(res, 'correlations') and res.correlations:
            fig2, ax2 = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
            ax2.set_facecolor('#1a1c23')
            ax2.bar(range(len(res.correlations)), res.correlations, color="#10b981" if res.found else "#ef4444")
            ax2.axhline(res.threshold, color="white", linestyle="--", alpha=0.5)
            ax2.set_title("Geometric Correlation per Bit", color='white', fontsize=10)
            ax2.tick_params(colors='white', labelsize=8)
            st.pyplot(fig2)
        elif wm_audio is not None:
             st.info("📊 Detection required to show correlation per bit.")

    with v2:
        if audio is not None:
            fig3, ax3 = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
            ax3.set_facecolor('#1a1c23')
            N = min(len(audio), 2**14)
            fft = np.abs(np.fft.rfft(audio[:N]))
            freqs = np.fft.rfftfreq(N, 1/sr)
            ax3.plot(freqs, 20*np.log10(fft + 1e-6), color="#f59e0b", alpha=0.9)
            ax3.set_title("FFT Magnitude Spectrum (dB)", color='white', fontsize=10)
            ax3.set_xlim(0, 10000)
            ax3.grid(True, alpha=0.1, color='white')
            ax3.tick_params(colors='white', labelsize=8)
            st.pyplot(fig3)
        else:
            st.info("📉 Spectrum preview unavailable.")
        
        # Confidence Distribution
        if res and hasattr(res, 'correlations') and res.correlations:
            valid_corrs = np.array(res.correlations)
            valid_corrs = valid_corrs[np.isfinite(valid_corrs)]
            
            fig4, ax4 = plt.subplots(figsize=(8, 3.5), facecolor='#0e1117')
            ax4.set_facecolor('#1a1c23')
            if len(valid_corrs) > 0 and np.ptp(valid_corrs) > 1e-9:
                ax4.hist(valid_corrs, bins=25, color="#a855f7", alpha=0.7)
            else:
                ax4.text(0.5, 0.5, "Insufficient variance for histogram", 
                         ha='center', va='center', transform=ax4.transAxes, color='white')
            ax4.set_title("Confidence Statistics Distribution", color='white', fontsize=10)
            ax4.tick_params(colors='white', labelsize=8)
            st.pyplot(fig4)
        else:
             st.info("📉 Stats will appear after forensic analysis.")

# ===========================================================================
# ACT 1: THE MACHINE (FUNDAMENTALS + OPERATIONAL)
# ===========================================================================

def act1_the_machine():
    st.markdown("""
        <div class="hero-container">
            <h1 class="hero-title">🏠 Fundamentals Laboratory</h1>
            <p class="hero-subtitle">Interactive exploration of Vector Geometry, Correlation, and the "Geometry of Trust".</p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- A1: AUDIO INPUT ---
    st.markdown("### 🎙️ A1 — Source Signal Acquisition")
    sc1, sc2 = st.columns([1, 1])
    with sc1:
        min_sec = st.session_state.config.min_audio_seconds
        st.info(f"💡 Recommended duration: >{min_sec:.1f}s")
        rec_dur = st.slider("Recording Duration (s)", 2, 30, 15)
        if st.button("🎤 Record Source"):
            with st.spinner("Recording..."):
                fs = 44100
                rec = sd.rec(int(rec_dur * fs), samplerate=fs, channels=1)
                sd.wait()
                st.session_state.audio = to_canonical(rec.flatten(), fs, st.session_state.config)
                st.session_state.wm_audio = None
                st.session_state.attacked_audio = None
            st.success("Captured.")
    with sc2:
        uploaded = st.file_uploader("Upload Audio (WAV)", type=["wav"])
        if uploaded:
            data, sr_in = sf.read(uploaded)
            st.session_state.audio = to_canonical(data, sr_in, st.session_state.config)
            st.session_state.wm_audio = None
            st.session_state.attacked_audio = None

    if st.session_state.audio is None:
        st.warning("Please record or upload audio to begin the laboratory.")
        return

    # --- FUNDAMENTALS LAB SECTIONS ---
    st.markdown("---")
    st.header("🧪 Fundamentals Laboratory")
    
    # SECTION A: VECTOR GEOMETRY
    st.markdown("""
        <div class="lab-card card-accent-blue">
            <div class="card-header">
                <span class="card-icon">📐</span>
                <p class="card-title">SECTION A — Audio → Vector Geometry</p>
            </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.subheader("From Sound to Points in Space")
        c1, c2 = st.columns(2)
        with c1:
            cfg = st.session_state.config
            frame = st.session_state.audio[:cfg.frame_size]
            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(cfg.frame_size, 1/44100)
            
            # Extract bins
            v = fft[cfg.bin_low:cfg.bin_low + 16] # Take 16 for demo
            st.write("**Extracted Magnitude Vector (v):**")
            st.dataframe(v.reshape(1, -1), hide_index=True)
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(v)), v, color="#1f77b4")
            ax.set_title("Visualizing v (Bins 1k-1.5kHz)")
            st.pyplot(fig)
        with c2:
            st.markdown("""
            **What are you seeing?**
            We've taken a slice of your audio's spectrum. Each bar is a 'feature' of your voice. 
            In watermarking, we treat these bars as a multi-dimensional **Vector**. 
            To hide data, we will nudge these bars slightly according to a secret pattern.
            """)
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION B & C: THE CORRELATION SANDBOX
    st.markdown("""
        <div class="lab-card card-accent-green">
            <div class="card-header">
                <span class="card-icon">🎯</span>
                <p class="card-title">SECTION B/C — The Correlation Sandbox</p>
            </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.subheader("Interactive Geometry: Key Alignment vs. Orthogonality")
        
        # 1. Math Explanation
        st.latex(r"C = \frac{1}{M} \sum_{i=1}^{M} v_i \cdot p_i")
        st.markdown(r"""
        **What is Geometric Correlation?**
        Think of your audio $(\mathbf{v})$ and your secret key $(\mathbf{p})$ as arrows in space. 
        Correlation $(\mathbf{C})$ measures how much the audio arrow 'points' in the same direction as the key arrow.
        """)

        sand1, sand2 = st.columns(2)
        with sand1:
            st.markdown("### 🔑 Side 1: The Secret Key (Owner)")
            owner_id_lab = st.text_input("Set Secret Owner ID", st.session_state.owner_id, key="sandbox_owner")
            oid_bytes_lab = ensure_8_bytes(owner_id_lab)
            st.info("🔐 The Owner ID is hashed (SHA256) to create a deterministic seed. This ensures that every ID produces a unique, fixed pattern of +/- 1.")
            p_secret = generate_pn_sequence(oid_bytes_lab, 0, cfg.chips_per_bit)[:16]
            
            st.caption("This ID generates the specific +/-1 pattern we will hide.")
            fig_p1, ax_p1 = plt.subplots(figsize=(5, 1.5))
            ax_p1.bar(range(16), p_secret, color="red", alpha=0.6)
            ax_p1.set_title("Secret Pattern (p)")
            st.pyplot(fig_p1)

        with sand2:
            st.markdown("### 🕵️ Side 2: The Detector Key (Probe)")
            probe_id = st.text_input("Enter Probe ID to check", "STRANGER_X", key="sandbox_probe")
            pid_bytes = ensure_8_bytes(probe_id)
            p_probe = generate_pn_sequence(pid_bytes, 0, cfg.chips_per_bit)[:16]
            
            st.caption("Try to 'guess' the owner ID to see if correlation works.")
            fig_p2, ax_p2 = plt.subplots(figsize=(5, 1.5))
            ax_p2.bar(range(16), p_probe, color="purple", alpha=0.6)
            ax_p2.set_title("Probe Pattern (p')")
            st.pyplot(fig_p2)

        st.markdown("---")
        # 2. Toy Embedding Nudge
        st.write("### 🏗️ Step 2: Hiding the Secret Pattern")
        alpha_toy = st.slider("Strength of Hidden Pattern (alpha)", 0.0, 2.0, 0.0, help="Nudge the audio vector towards the Secret Key.")
        
        # Modify v for sandbox visualization
        v_mod = v + alpha_toy * p_secret
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            c_sec = np.dot(v_mod, p_secret) / len(v_mod)
            st.metric("Correlation with SECRET", f"{c_sec:.4f}")
            if c_sec > 0.5: 
                st.success("✅ Pattern Found!")
                st.caption("The audio arrow is now aligned with the Secret Key.")
        with col_res2:
            c_prb = np.dot(v_mod, p_probe) / len(v_mod)
            st.metric("Correlation with PROBE", f"{c_prb:.4f}")
            if abs(c_prb) < 0.2: 
                st.info("❌ Unauthorized Key (Orthogonal)")
                st.caption("Notice this value stays near 0 even if you increase α! The keys are mathematically independent.")

        # 3. 2D Comparison Plot
        st.write("### 📐 Step 3: Visualizing the 'Arrows'")
        
        # Projections for 2D
        # Normalized for plotting
        v_plot = v_mod[:2] / (np.max(np.abs(v_mod[:2])) + 1e-6)
        p_sec_plot = p_secret[:2]
        p_prb_plot = p_probe[:2]
        
        fig_dual, ax_dual = plt.subplots(figsize=(6, 6))
        ax_dual.set_xlim(-1.5, 1.5); ax_dual.set_ylim(-1.5, 1.5)
        ax_dual.axhline(0, color='gray', linestyle='--', alpha=0.3)
        ax_dual.axvline(0, color='gray', linestyle='--', alpha=0.3)
        
        # Secret Key Arrow
        ax_dual.arrow(0, 0, p_sec_plot[0], p_sec_plot[1], head_width=0.08, color='red', label='Secret Key (p)', zorder=10)
        # Probe Key Arrow
        ax_dual.arrow(0, 0, p_prb_plot[0], p_prb_plot[1], head_width=0.08, color='purple', label='Probe Key (p\')', zorder=10)
        # Audio Arrow
        ax_dual.arrow(0, 0, v_plot[0], v_plot[1], head_width=0.08, color='blue', label='Audio Vector (v)', width=0.02)
        
        ax_dual.legend(loc="upper right")
        ax_dual.set_title("The Geometry of Trust")
        st.pyplot(fig_dual)

        # AI Analysis for Sandbox
        if st.button("🤖 Explain Sandbox Geometry", key="ai_sandbox_btn"):
            with st.spinner("Analyzing Geometry..."):
                sandbox_metrics = {'snr': 'N/A', 'conf': c_sec, 'sec_corr': c_sec, 'prb_corr': c_prb}
                explanation = openai_explanation(sandbox_metrics, context="Correlation Sandbox (Geometric Alignment)")
                st.session_state.ai_explanation = explanation
                st.rerun()

        st.markdown(f"""
        **The Lesson of Dimension:**
        1. **Alignment**: As you increase $\alpha$, notice how the **Blue arrow (Audio)** slowly tips toward the **Red arrow (Secret Key)**. This is embedding.
        2. **Independence**: Even when the audio is aligned with Red, it remains almost perpendicular (orthogonal) to the **Purple arrow (Probe ID)**.
        3. **Failure**: This is why unauthorized users see near-zero correlation. In a high-dimensional space (256+), random arrows are almost always at 90 degrees to each other.
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION D: FLOWCHART
    st.markdown("""
        <div class="lab-card card-accent-purple">
            <div class="card-header">
                <span class="card-icon">⛙</span>
                <p class="card-title">SECTION D — Watermark Generation Pipeline</p>
            </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.graphviz_chart("""
        digraph {
            rankdir=LR;
            ID [label="Owner ID"];
            SHA [label="SHA256(owner_id || bit_index)"];
            Seed [label="Secret Seed"];
            RNG [label="PRNG"];
            PN [label="PN Sequence +/-1"];
            Bits [label="Data Bits"];
            Map [label="Bit Mapping"];
            FFT [label="Original FFT"];
            Mod [label="Magnitude Modulation"];
            IFFT [label="Time Domain Audio"];
            
            ID -> SHA -> Seed -> RNG -> PN;
            Bits -> Map -> PN;
            PN -> Mod;
            FFT -> Mod -> IFFT;
        }
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION E: CRYPTO LAB
    st.markdown("""
        <div class="lab-card card-accent-amber">
            <div class="card-header">
                <span class="card-icon">🔐</span>
                <p class="card-title">SECTION E — Cryptographic Ownership Lab</p>
            </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.subheader("Proving it's Yours")
        mode = st.radio("Lab Mode", ["Owner (Sign)", "Verifier (Check)"], horizontal=True)
        h_val = audio_hash(st.session_state.audio).hex()[:16]
        
        if mode == "Owner (Sign)":
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.write(f"Audio Hash: `0x{h_val}...`")
                user = st.session_state.users["Owner"]
                st.code(user['pub'].public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()[:50] + "...", language="text")
                if st.checkbox("Reveal Private Key"):
                    st.warning("Keep this secret! It represents your legal identity.")
                    st.code("ED25519_PRIV_KEY_****************", language="text")
            with c_e2:
                sig = sign(user['priv'], h_val.encode())
                st.session_state.last_sig = sig
                st.success("Message Signed.")
                st.write("Signature (Hex):")
                st.code(sig.hex()[:64], line_numbers=True)
        else:
            c_v1, c_v2 = st.columns(2)
            with c_v1:
                tester = st.selectbox("Testing Persona", ["Owner", "Adversary"])
                pub_test = st.session_state.users[tester]['pub']
                st.info(f"Using {st.session_state.users[tester]['name']}'s Public Key")
            with c_v2:
                if 'last_sig' in st.session_state:
                    is_valid = verify(pub_test, h_val.encode(), st.session_state.last_sig)
                    if is_valid: st.success("✅ SIGNATURE VALID")
                    else: st.error("❌ SIGNATURE FAILURE: UNTRUSTED ENTITY")
                else: st.info("Sign audio in Owner Mode first.")
    st.markdown("</div>", unsafe_allow_html=True)

    # SECTION F & G: SHA & AVALANCHE
    st.markdown("""
        <div class="lab-card card-accent-ruby">
            <div class="card-header">
                <span class="card-icon">🌌</span>
                <p class="card-title">SECTION F/G — The SHA256 Universe & Avalanche Effect</p>
            </div>
    """, unsafe_allow_html=True)
    with st.container():
        st.subheader("The Unbreakable Fingerprint")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("**SHA256 Collision Visualization**")
            space_size = st.select_slider("Hash Space Modulo", options=[10, 100, 1000, 10000], value=100)
            st.write("Explaining $P \approx 1/2^{256}$ through scale.")
            # Simulation
            hashes = [hash(str(i)) % space_size for i in range(100)]
            unique = len(set(hashes))
            collisions = 100 - unique
            st.metric("Collision Count (per 100 items)", collisions, delta="High Risk" if collisions > 10 else "Low Risk")
        with col_f2:
            if st.button("🚨 Avalanche Test: Change 1 Sample"):
                st.info("Original Hash: " + h_val)
                modified = st.session_state.audio.copy()
                modified[100] += 0.0001
                new_h = audio_hash(modified).hex()[:16]
                st.error("Modified Hash: " + new_h)
                st.caption("Changing even 0.01% of the data creates a completely new identity.")
    st.markdown("</div>", unsafe_allow_html=True)

    # --- OPERATIONAL SEGMENT ---
    st.markdown("---")
    st.header("⚙️ Operational Control")
    st.info("Now that intuition is built, let's run the real embedding engine.")
    f_op1, f_op2 = st.columns(2)
    with f_op1:
        st.subheader("2. Embedding Laboratory")
        cfg = st.session_state.config
        if st.session_state.audio is not None:
             audio_sec = len(st.session_state.audio) / 44100
             if audio_sec < cfg.min_audio_seconds:
                 st.error(f"⚠️ Audio too short ({audio_sec:.1f}s). Need ≥{cfg.min_audio_seconds:.1f}s.")
             else:
                 st.success(f"✅ Audio length OK ({audio_sec:.1f}s)")

        if st.button("✨ Embed Invisible Watermark", key="act1_embed"):
            if st.session_state.audio is not None and len(st.session_state.audio) / 44100 < cfg.min_audio_seconds:
                st.error("Cannot embed: Audio is too short.")
            else:
                with st.spinner("Embedding..."):
                    user = st.session_state.users["Owner"]
                    oid_bytes = ensure_8_bytes(st.session_state.owner_id)
                    st.session_state.wm_audio = embed(st.session_state.audio, oid_bytes, user['priv'], cfg)
                    snr = calculate_snr(st.session_state.audio, st.session_state.wm_audio)
                    st.session_state.metrics_act1.update({
                        'snr': snr,
                        'bits': st.session_state.config.total_bits,
                        'sig': None, 'conf': 0.0, 'ecc': 0
                    })
                    st.success(f"Watermark hidden. SNR: {snr:.2f} dB")
        if st.session_state.wm_audio is not None:
            st.audio(get_audio_path(st.session_state.wm_audio, 44100, "act1_wm", "watermarked"), format="audio/wav")

    with f_op2:
        st.subheader("3. Robustness Challenges")
        if st.session_state.wm_audio is not None:
            atk_type = st.selectbox("Apply Attack", list(ATTACK_REGISTRY.keys()), key="act1_atk")
            atk_p = st.slider("Attack Parameter", 0.0, 1.0, 0.2, key="act1_atkp")
            if st.button("🔥 Execute Attack"):
                st.session_state.attacked_audio = apply_attack(atk_type, st.session_state.wm_audio, atk_p)
                st.session_state.recovery_msg_act1 = None # Clear old recovery status
                st.success("Audio Distortion Applied.")
            if st.session_state.attacked_audio is not None:
                st.audio(get_audio_path(st.session_state.attacked_audio, 44100, "act1_atk", "attacked"), format="audio/wav")
            
            if st.button("🔍 Try Recovery from Attack", type="primary", key="act1_recovery_btn"):
                with st.spinner("Attempting Forensic Recovery..."):
                    pub = st.session_state.users["Owner"]["pub"]
                    oid_bytes = ensure_8_bytes(st.session_state.owner_id)
                    res = detect(st.session_state.attacked_audio, oid_bytes, pub, st.session_state.config)
                    st.session_state.res_act1 = res
                    st.session_state.metrics_act1.update({
                        'conf': res.mean_confidence if hasattr(res, 'mean_confidence') else 0.0,
                        'bits': len(res.payload) if hasattr(res, 'payload') else 0,
                        'sig': res.signature_valid if hasattr(res, 'signature_valid') else False,
                        'ecc': res.ecc_stats.get('corrected', 0) if hasattr(res, 'ecc_stats') else 0,
                        'snr': calculate_snr(st.session_state.audio, st.session_state.wm_audio) # Act 1 specific
                    })
                    if res.found:
                        st.session_state.recovery_msg_act1 = "✅ Watermark Recovered despite attack!"
                        st.session_state.recovery_type_act1 = "success"
                    else:
                        st.session_state.recovery_msg_act1 = "❌ Recovery Failed: Signal too distorted."
                        st.session_state.recovery_type_act1 = "error"
            
            if st.session_state.recovery_msg_act1:
                if st.session_state.recovery_type_act1 == "success":
                    st.success(st.session_state.recovery_msg_act1)
                else:
                    st.error(st.session_state.recovery_msg_act1)

    # --- ALWAYS VISIBLE PLOTS ---
    st.markdown("---")
    standard_visualizations(st.session_state.audio, 44100, "Source Visuals", 
                            st.session_state.wm_audio, 
                            res=st.session_state.res_act1)
    standard_metric_panel(st.session_state.metrics_act1, panel_id="act1_main")

# ===========================================================================
# EXTRA WINDOWS (ACT 2-4)
# ===========================================================================

def extra_embedding_studio():
    st.markdown("""
        <div class="hero-container" style="border-left-color: #3b82f6;">
            <h1 class="hero-title">⚡ Embedding Studio</h1>
            <p class="hero-subtitle">High-precision watermark injection and real-time spectrum auditing.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.session_state.audio is None:
        st.warning("Load audio in Act 1 first.")
        return
        
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("---")
        st.header("⚙️ Parameters")
        toy = st.toggle("🧸 Toy Mode (High Visibility)", value=False)
        if toy:
            st.session_state.config.embed_strength = 0.8
            st.session_state.config.chips_per_bit = 31
            st.info("Toy Mode: Forced high α and low M for visual clarity.")
        else:
            # Global parameters removed here, now in sidebar
            st.info("💡 Use the Sidebar to adjust α and Chips.")
        
        min_sec = st.session_state.config.min_audio_seconds
        st.info(f"💾 Param Requirement: ≥{min_sec:.1f}s")
        if st.session_state.audio is not None:
            audio_sec = len(st.session_state.audio)/44100
            if audio_sec < min_sec:
                st.error(f"❌ Audio too short ({audio_sec:.1f}s)")
            else:
                st.success(f"✅ Audio length OK ({audio_sec:.1f}s)")
        
        if st.button("💾 Finalize & Save", type="primary"):
            if len(st.session_state.audio) / 44100 < st.session_state.config.min_audio_seconds:
                st.error("Audio too short for current parameters.")
            else:
                user = st.session_state.users["Owner"]
                oid_bytes = ensure_8_bytes(st.session_state.owner_id)
                st.session_state.wm_audio = embed(st.session_state.audio, oid_bytes, user['priv'], st.session_state.config)
                snr = calculate_snr(st.session_state.audio, st.session_state.wm_audio)
                st.session_state.metrics_studio.update({
                    'snr': snr,
                    'bits': st.session_state.config.total_bits,
                    'sig': None, 'conf': 0.0, 'ecc': 0
                })
                path = get_audio_path(st.session_state.wm_audio, 44100, f"studio_{int(time.time())}", "watermarked")
                st.success(f"Archived to `{path}`")
            
    with c2:
        standard_visualizations(st.session_state.audio, 44100, "Real-Time Diff Spectrum", st.session_state.wm_audio)
        standard_metric_panel(st.session_state.metrics_studio, panel_id="studio_embed")

def extra_detection_studio():
    st.markdown("""
        <div class="hero-container" style="border-left-color: #ef4444;">
            <h1 class="hero-title">🔍 Detection & Verification</h1>
            <p class="hero-subtitle">Forensic scanning and signature validation for suspected audio evidence.</p>
        </div>
    """, unsafe_allow_html=True)
    files = [f for f in os.listdir("watermarked") if f.endswith(".wav")]
    if not files:
        st.info("No watermarked files found. Create one in Act 1 or Embedding Studio.")
        return
        
    dc1, dc2 = st.columns([1, 1.5])
    with dc1:
        st.subheader("🕵️ Forensic Configuration")
        selection = st.selectbox("Select Evidence", files)
        
        # Explicit Parameter Check
        st.markdown(f"**Scan Depth:** `{st.session_state.config.chips_per_bit} Chips/Bit`")
        st.caption("⚠️ This MUST match the setting used during embedding (default 63).")
        
        detect_id = st.text_input("Claimed Owner ID", st.session_state.owner_id)
        st.caption("The secret key that generates the unique PN pattern.")

        if st.button("🔍 Run Forensic Scan", type="primary", use_container_width=True):
            data, sr = sf.read(os.path.join("watermarked", selection))
            pub = st.session_state.users["Owner"]['pub']
            oid_bytes = ensure_8_bytes(detect_id)
            
            with st.spinner("Scanning for hidden pattern..."):
                res = detect(data, oid_bytes, pub, st.session_state.config)
                st.session_state.res_detect = res
                st.session_state.detect_data = data
                st.session_state.metrics_detect.update({
                    'conf': getattr(res, 'mean_confidence', 0.0),
                    'bits': len(getattr(res, 'payload', b"")),
                    'sig': getattr(res, 'signature_valid', False),
                    'ecc': getattr(res, 'ecc_stats', {}).get('corrected', 0),
                    'found': getattr(res, 'found', False)
                })
                
                if getattr(res, 'found', False):
                    st.success("✅ Forensic Match: Watermark Found!")
                else:
                    st.error("❌ No Match: Watermark not found.")
                    st.info("💡 **Why did it fail?**\n1. Check if 'Chips per Bit' matches the embedding setting.\n2. Ensure the 'Claimed Owner ID' is exactly what Alice used.\n3. Increase Scan Depth if the audio is heavily distorted.")

    with dc2:
        if st.session_state.res_detect is not None and st.session_state.detect_data is not None:
            res = st.session_state.res_detect
            data = st.session_state.detect_data
            standard_visualizations(data, 44100, "Forensic Extraction", res=res)
            standard_metric_panel(st.session_state.metrics_detect, panel_id="studio_detect")

def extra_attack_lab():
    st.markdown("""
        <div class="hero-container" style="border-left-color: #f59e0b;">
            <h1 class="hero-title">🛡️ Adversarial Attack Lab</h1>
            <p class="hero-subtitle">Stress-testing watermark robustness against common signal processing distortion.</p>
        </div>
    """, unsafe_allow_html=True)
    files = [f for f in os.listdir("watermarked") if f.endswith(".wav")]
    if not files:
        st.info("No watermarked files found.")
        return
        
    ac1, ac2 = st.columns([1, 2])
    with ac1:
        selection = st.selectbox("Select Victim", files, key="atk_select")
        atk_type = st.selectbox("Attack", list(ATTACK_REGISTRY.keys()), key="atk_type")
        sev = st.slider("Severity", 0.0, 1.0, 0.2, key="atk_sev")
        
        if st.button("🔥 Distort", key="atk_go"):
            data, sr = sf.read(os.path.join("watermarked", selection))
            st.session_state.attacked_audio = apply_attack(atk_type, data, sev)
            st.session_state.recovery_msg_atk = None # Clear old recovery status
            st.success("Destruction Complete.")
            
    with ac2:
        if st.session_state.attacked_audio is not None:
            st.subheader("Degradation View")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(st.session_state.attacked_audio[:2000], color="red", alpha=0.5)
            ax.set_title("Distorted Waveform")
            st.pyplot(fig)
            st.audio(get_audio_path(st.session_state.attacked_audio, 44100, "studio_atk", "attacked"), format="audio/wav")

            if st.button("🔍 Try Recovery from Attack", type="primary", key="studio_atk_recovery"):
                with st.spinner("Attempting Forensic Recovery..."):
                    pub = st.session_state.users["Owner"]["pub"]
                    oid_bytes = ensure_8_bytes(st.session_state.owner_id)
                    res = detect(st.session_state.attacked_audio, oid_bytes, pub, st.session_state.config)
                    st.session_state.res_atk = res
                    st.session_state.metrics_atk.update({
                        'conf': res.mean_confidence if hasattr(res, 'mean_confidence') else 0.0,
                        'bits': len(res.payload) if hasattr(res, 'payload') else 0,
                        'sig': res.signature_valid if hasattr(res, 'signature_valid') else False,
                        'ecc': res.ecc_stats.get('corrected', 0) if hasattr(res, 'ecc_stats') else 0,
                    })
                    if res.found:
                        st.session_state.recovery_msg_atk = "✅ Watermark Recovered!"
                        st.session_state.recovery_type_atk = "success"
                    else:
                        st.session_state.recovery_msg_atk = "❌ Recovery Failed."
                        st.session_state.recovery_type_atk = "error"

            if st.session_state.recovery_msg_atk:
                if st.session_state.recovery_type_atk == "success":
                    st.success(st.session_state.recovery_msg_atk)
                else:
                    st.error(st.session_state.recovery_msg_atk)
                    
            if st.session_state.res_atk:
                st.markdown("---")
                standard_visualizations(st.session_state.attacked_audio, 44100, "Attack Forensics", res=st.session_state.res_atk)
                standard_metric_panel(st.session_state.metrics_atk, panel_id="studio_atk")

def extra_sha256_lab():
    st.markdown("""
        <div class="hero-container" style="border-left-color: #a855f7;">
            <h1 class="hero-title">🧬 SHA-256 & Pattern Lab</h1>
            <p class="hero-subtitle">Visualizing the cryptographic DNA that maps identity keys to orthogonal patterns.</p>
        </div>
    """, unsafe_allow_html=True)
    
    id_a = st.text_input("Owner ID A (Alice)", st.session_state.owner_id)
    id_b = st.text_input("Owner ID B (Bob)", "EVIL_BOB_99")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("🔢 Binary Avalanche Mapping")
        # 1. Hashing
        hash_a = hashlib.sha256(id_a.encode()).hexdigest()
        hash_b = hashlib.sha256(id_b.encode()).hexdigest()
        
        c1, c2 = st.columns(2)
        c1.markdown(f"**SHA-256 (Alice):**")
        c1.code(f"{hash_a[:32]}\n{hash_a[32:]}")
        c2.markdown(f"**SHA-256 (Bob):**")
        c2.code(f"{hash_b[:32]}\n{hash_b[32:]}")
        
        # 2. Bits
        bits_a = bin(int(hash_a, 16))[2:].zfill(256)
        bits_b = bin(int(hash_b, 16))[2:].zfill(256)
        
        st.markdown("**First 128 bits converted to Pattern Components (-1, +1):**")
        # Visual bit stream comparison
        st.code(f"Alice: {bits_a[:128]}\nBob:   {bits_b[:128]}")
        st.info("💡 Even a single character change in the ID causes a ~50% bit flip (The Avalanche Effect).")

    with col2:
        st.subheader("📊 Statistical Orthogonality")
        
        # 3. PN Sequence (+1, -1) for orthogonality check
        pn_a = np.array([1 if b == '1' else -1 for b in bits_a])
        pn_b = np.array([1 if b == '1' else -1 for b in bits_b])
        
        # Dot Product (Normalization by length)
        dot = np.dot(pn_a, pn_b)
        overlap = np.sum(np.array(list(bits_a)) == np.array(list(bits_b)))
        
        mc1, mc2 = st.columns(2)
        mc1.metric("Bit Overlap (Bits)", f"{overlap}/256")
        mc2.metric("Orthogonality (Dot)", f"{dot}", help="Closer to 0 means better security/independence.")
        
        # Overlap "Venn" Visualization
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0e1117')
        ax.set_facecolor('#0e1117')
        
        circ_a = plt.Circle((0.4, 0.5), 0.35, color='#1f77b4', alpha=0.4, label='Alice ID')
        circ_b = plt.Circle((0.6, 0.5), 0.35, color='#ff7f0e', alpha=0.4, label='Bob ID')
        ax.add_patch(circ_a)
        ax.add_patch(circ_b)
        
        overlap_pct = (overlap/256) * 100
        ax.text(0.5, 0.5, f"{overlap}/256 Bits\nCommon", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.legend(loc='lower center', facecolor='#262730', labelcolor='white')
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("🌀 From Bits to Sound Wave")
    st.markdown(r"""
    1. **Text ID** $\rightarrow$ **SHA-256 Hash** (256 bits).
    2. **Hash Bits** $\rightarrow$ **PN Sequence** (Seeded random +1 / -1 jumps).
    3. **PN Sequence** $\times$ **Alpha ($\alpha$)** $\rightarrow$ **The Watermark Pattern**.
    4. **Audio Signal** $+$ **Pattern** $\approx$ **Audio Signal**.
    
    Because Key A and Key B are **orthogonal** (their patterns don't align), Bob cannot detect Alice's signature, even if he knows the algorithm!
    """)

# --- Routing ---
global_sidebar()
tab_list = st.tabs(["🏠 Act 1", "🧬 SHA-256 Lab", "⚡ Studio", "🔍 Detect", "🛡️ Attacks"])

with tab_list[0]: act1_the_machine()
with tab_list[1]: extra_sha256_lab()
with tab_list[2]: extra_embedding_studio()
with tab_list[3]: extra_detection_studio()
with tab_list[4]: extra_attack_lab()
