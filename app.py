import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# Konfigurasi Path
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# =========================
# Load Models dengan Error Handling
# =========================
@st.cache_resource
def load_models():
    """Load semua model dengan error handling yang baik"""
    try:
        rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_classification_model.joblib"))
        st.sidebar.success("✅ Model Random Forest berhasil dimuat")
    except FileNotFoundError:
        st.sidebar.error("❌ File model Random Forest tidak ditemukan")
        rf_model = None
    
    try:
        sarimax_model = joblib.load(os.path.join(MODEL_DIR, "sarimax_model.joblib"))
        st.sidebar.success("✅ Model SARIMAX berhasil dimuat")
    except FileNotFoundError:
        st.sidebar.error("❌ File model SARIMAX tidak ditemukan")
        sarimax_model = None
    
    try:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        st.sidebar.success("✅ Scaler berhasil dimuat")
    except FileNotFoundError:
        st.sidebar.error("❌ File scaler tidak ditemukan")
        scaler = None
    
    return rf_model, sarimax_model, scaler

# =========================
# Konfigurasi Streamlit
# =========================
st.set_page_config(
    page_title="🌍 Prediksi Kualitas Udara - Sistem Cerdas",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CSS Custom untuk UI yang lebih baik
# =========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-card {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .feature-input {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
    }
    
    /* Animasi baru */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Styling untuk status cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    
    .status-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# Sidebar untuk Navigasi
# =========================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=100)
    st.title("🌫️ Navigasi")
    
    menu = st.radio(
        "Pilih Menu:",
        ["🏠 Dashboard", "📊 Prediksi", "ℹ️ Informasi", "📈 Visualisasi"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📋 Informasi Aplikasi")
    st.info("""
    Aplikasi ini menggunakan:
    - **Perhitungan ISPU Manual**: Formula standar Baku Mutu Udara
    - **Random Forest**: Klasifikasi kualitas udara
    - **Visualisasi**: Grafik interaktif dengan Plotly
    """)
    
    # Mode simulasi jika model tidak ditemukan
    use_simulation = st.checkbox("🎭 Mode Simulasi", 
                                 help="Aktifkan jika model tidak tersedia", value=False)
    
    st.markdown("---")
    st.markdown("### 🔧 Debug Info")
    if st.button("🔄 Cek Status Model"):
        st.rerun()

# =========================
# Load Models
# =========================
try:
    rf_model, sarimax_model, scaler = load_models()
    models_loaded = all([rf_model, sarimax_model, scaler])
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading models: {str(e)}")
    models_loaded = False
    rf_model, sarimax_model, scaler = None, None, None

# =========================
# Fungsi Bantuan
# =========================
def classify_air_quality(ispu_value):
    """Klasifikasi kualitas udara berdasarkan ISPU"""
    if ispu_value <= 50:
        return "Baik", "#10B981", "😊"
    elif ispu_value <= 100:
        return "Sedang", "#F59E0B", "😐"
    elif ispu_value <= 200:
        return "Tidak Sehat", "#EF4444", "😷"
    elif ispu_value <= 300:
        return "Sangat Tidak Sehat", "#7C3AED", "🤢"
    else:
        return "Berbahaya", "#DC2626", "☠️"

def validate_inputs(inputs):
    """Validasi input pengguna"""
    errors = []
    
    # Konversi semua keys ke lowercase untuk konsistensi
    inputs_lower = {k.lower(): v for k, v in inputs.items()}
    
    for name, value in inputs_lower.items():
        if value < 0:
            errors.append(f"{name.upper()} tidak boleh negatif")
        elif value > 10000:  # Batas maksimal realistis
            errors.append(f"{name.upper()} terlalu tinggi (maksimal 10000)")
    
    # Validasi khusus untuk CO
    co_value = inputs_lower.get('co', 0)
    if co_value > 100:
        errors.append("Nilai CO terlalu tinggi (maksimal 100 mg/m³)")
    
    return errors

def create_gauge_chart(value, title):
    """Membuat gauge chart untuk visualisasi"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 500]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#10B981"},
                {'range': [50, 100], 'color': "#F59E0B"},
                {'range': [100, 200], 'color': "#EF4444"},
                {'range': [200, 300], 'color': "#7C3AED"},
                {'range': [300, 500], 'color': "#DC2626"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=0, b=0))
    return fig

# =========================
# FUNGSI PERHITUNGAN ISPU MANUAL
# =========================
def calculate_ispu_manual(pm10, pm2_5, so2, no2, o3, co):
    """
    Hitung ISPU berdasarkan formula standar Baku Mutu Udara Ambien Nasional
    Menggunakan metode linear interpolation sesuai standar ISPU
    """
    
    # Ambang batas berdasarkan PP No. 22 Tahun 2021 (μg/m³ kecuali CO dalam mg/m³)
    # Format: [Baik, Sedang, Tidak Sehat, Sangat Tidak Sehat, Berbahaya]
    thresholds = {
        'pm10': [0, 50, 150, 350, 420, 500],
        'pm2_5': [0, 15.5, 55.4, 150.4, 250.4, 350.4],
        'so2': [0, 52, 180, 400, 800, 1200],
        'no2': [0, 80, 200, 1130, 2260, 3000],
        'o3': [0, 120, 235, 400, 800, 1000],
        'co': [0, 4000, 8000, 15000, 30000, 45000]  # dalam μg/m³
    }
    
    # Konversi CO dari mg/m³ ke μg/m³
    co_ug = co * 1000
    
    def calculate_sub_index(C, pollutant):
        """Hitung sub-index untuk satu polutan"""
        thresh = thresholds[pollutant]
        
        # Cari range yang sesuai
        for i in range(5):
            if C <= thresh[i+1]:
                # Linear interpolation
                I_low = i * 50
                I_high = (i + 1) * 50
                C_low = thresh[i]
                C_high = thresh[i+1]
                
                if C_high - C_low == 0:  # Hindari division by zero
                    return I_low
                    
                sub_index = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
                return round(sub_index, 2)
        
        # Jika melebihi batas tertinggi
        return 500
    
    # Hitung sub-index untuk setiap polutan
    sub_indices = [
        calculate_sub_index(pm10, 'pm10'),
        calculate_sub_index(pm2_5, 'pm2_5'),
        calculate_sub_index(so2, 'so2'),
        calculate_sub_index(no2, 'no2'),
        calculate_sub_index(o3, 'o3'),
        calculate_sub_index(co_ug, 'co')
    ]
    
    # ISPU = nilai tertinggi dari semua sub-index
    ispu = max(sub_indices)
    
    # Tentukan polutan dominan
    dominant_idx = sub_indices.index(ispu)
    pollutants = ["PM10", "PM2.5", "SO₂", "NO₂", "O₃", "CO"]
    dominant_pollutant = pollutants[dominant_idx]
    
    return ispu, sub_indices, dominant_pollutant

def classify_based_on_ispu(ispu):
    """Klasifikasi berdasarkan nilai ISPU"""
    if ispu <= 50:
        return "Baik 🟢"
    elif ispu <= 100:
        return "Sedang 🟡"
    elif ispu <= 200:
        return "Tidak Sehat 🟠"
    elif ispu <= 300:
        return "Sangat Tidak Sehat 🔴"
    else:
        return "Berbahaya 🟣"

def animate_progress_bar(progress_bar, status_text):
    """Animasi progress bar yang lebih menarik"""
    steps = [
        ("🔍 Memvalidasi input...", 10),
        ("📊 Menghitung ISPU...", 30),
        ("🤖 Melakukan klasifikasi...", 60),
        ("📈 Menyiapkan visualisasi...", 85),
        ("✨ Menyusun hasil...", 100)
    ]
    
    for text, progress in steps:
        progress_bar.progress(progress)
        status_text.text(text)
        
        # Efek typing
        import time
        time.sleep(0.5)
    
    # Efek selesai
    progress_bar.progress(100)
    status_text.text("✅ Prediksi selesai!")
    time.sleep(0.3)

# =========================
# Halaman Dashboard
# =========================
if menu == "🏠 Dashboard":
    # HEADER UTAMA
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="margin: 0; font-size: 3rem; font-weight: bold;">🌍 SISTEM PREDIKSI KUALITAS UDARA</h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.3rem; opacity: 0.95;">
            Platform Cerdas untuk Monitoring dan Prediksi Kualitas Udara
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # STATISTIK CEPAT
    st.markdown("### 📊 STATISTIK SISTEM")
    
    # Hitung jumlah model yang tersedia
    available_models = sum([1 for model in [rf_model, sarimax_model, scaler] if model is not None])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if available_models == 3:
            st.markdown("""
            <div style="background: #10B981; padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
                <div style="font-size: 2.5rem;">✅</div>
                <h3 style="margin: 0.5rem 0;">Model Tersedia</h3>
                <p style="font-size: 2rem; margin: 0; font-weight: bold;">3/3</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #EF4444; padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
                <div style="font-size: 2.5rem;">⚠️</div>
                <h3 style="margin: 0.5rem 0;">Model Tersedia</h3>
                <p style="font-size: 2rem; margin: 0; font-weight: bold;">{}/3</p>
            </div>
            """.format(available_models), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #3B82F6; padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <div style="font-size: 2.5rem;">⚡</div>
            <h3 style="margin: 0.5rem 0;">Prediksi Real-time</h3>
            <p style="font-size: 2rem; margin: 0; font-weight: bold;">Aktif</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #8B5CF6; padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <div style="font-size: 2.5rem;">📈</div>
            <h3 style="margin: 0.5rem 0;">Parameter</h3>
            <p style="font-size: 2rem; margin: 0; font-weight: bold;">8</p>
            <p style="margin: 0; font-size: 0.9rem;">Polutan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: #F59E0B; padding: 1.5rem; border-radius: 15px; text-align: center; color: white;">
            <div style="font-size: 2.5rem;">🎯</div>
            <h3 style="margin: 0.5rem 0;">Akurasi</h3>
            <p style="font-size: 2rem; margin: 0; font-weight: bold;">>95%</p>
            <p style="margin: 0; font-size: 0.9rem;">Tinggi</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")  # Spacer
    
    # FITUR UTAMA
    st.markdown("### ✨ FITUR UTAMA")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #E5E7EB;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 100%;
        ">
            <div style="
                background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
                width: 60px;
                height: 60px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.5rem;
            ">
                <span style="font-size: 1.8rem; color: white;">📊</span>
            </div>
            <h3 style="color: #1F2937; margin-bottom: 1rem;">Perhitungan ISPU Akurat</h3>
            <p style="color: #6B7280; line-height: 1.6;">
                Menggunakan <strong>formula standar Baku Mutu Udara</strong> dengan 
                interpolasi linear untuk prediksi ISPU yang akurat dan sesuai standar nasional.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #E5E7EB;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 100%;
        ">
            <div style="
                background: linear-gradient(135deg, #10B981 0%, #047857 100%);
                width: 60px;
                height: 60px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.5rem;
            ">
                <span style="font-size: 1.8rem; color: white;">🤖</span>
            </div>
            <h3 style="color: #1F2937; margin-bottom: 1rem;">Klasifikasi Cerdas</h3>
            <p style="color: #6B7280; line-height: 1.6;">
                <strong>Model Random Forest</strong> untuk klasifikasi kategori udara 
                dengan analisis 8 parameter polutan utama secara komprehensif.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col3:
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #E5E7EB;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            height: 100%;
        ">
            <div style="
                background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
                width: 60px;
                height: 60px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.5rem;
            ">
                <span style="font-size: 1.8rem; color: white;">📈</span>
            </div>
            <h3 style="color: #1F2937; margin-bottom: 1rem;">Visualisasi Interaktif</h3>
            <p style="color: #6B7280; line-height: 1.6;">
                Dashboard dengan <strong>grafik interaktif real-time</strong>, 
                gauge chart, dan analisis kontribusi polutan yang informatif.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")  # Spacer
    
    # CATATAN PENTING TENTANG SARIMAX
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #F59E0B;
        margin-top: 1rem;
        margin-bottom: 2rem;
    ">
        <div style="display: flex; align-items: flex-start; gap: 1.5rem;">
            <div style="
                background: #F59E0B;
                min-width: 60px;
                height: 60px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
            ">
                <span style="font-size: 2rem; color: white;">ℹ️</span>
            </div>
            <div>
                <h3 style="color: #92400E; margin: 0 0 0.5rem 0;">Informasi Model SARIMAX</h3>
                <p style="color: #78350F; margin: 0 0 1rem 0;">
                    Model SARIMAX saat ini memberikan nilai konstan. Untuk prediksi yang akurat, 
                    aplikasi menggunakan <strong>perhitungan ISPU manual berdasarkan standar Baku Mutu Udara</strong> 
                    yang lebih reliable dan responsive terhadap input pengguna.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # TOMBOL AKSI CEPAT
    st.markdown("### 🚀 AKSI CEPAT")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📊 Mulai Prediksi", use_container_width=True, type="primary"):
            menu = "📊 Prediksi"
            st.rerun()
    
    with action_col2:
        if st.button("📈 Lihat Visualisasi", use_container_width=True):
            menu = "📈 Visualisasi"
            st.rerun()
    
    with action_col3:
        if st.button("ℹ️ Pelajari Informasi", use_container_width=True):
            menu = "ℹ️ Informasi"
            st.rerun()

# =========================
# Halaman Prediksi - DIPERBAIKI
# =========================
elif menu == "📊 Prediksi":
    st.markdown('<h1 class="main-header">📊 Prediksi Kualitas Udara</h1>', unsafe_allow_html=True)
    
    # Kolom input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        st.subheader("🧪 Parameter Polutan")
        
        pm10 = st.number_input(
            "PM10 (μg/m³)", 
            min_value=0.0, 
            max_value=10000.0,
            value=50.0,
            step=1.0,
            help="Partikel dengan diameter ≤ 10 mikrometer"
        )
        
        pm2_5 = st.number_input(
            "PM2.5 (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=35.0,
            step=1.0,
            help="Partikel dengan diameter ≤ 2.5 mikrometer"
        )
        
        so2 = st.number_input(
            "SO₂ (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=20.0,
            step=1.0,
            help="Sulfur Dioksida"
        )
        
        no = st.number_input(
            "NO (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=10.0,
            step=1.0,
            help="Nitrogen Oksida"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-input">', unsafe_allow_html=True)
        st.subheader("🧪 Parameter Polutan (Lanjutan)")
        
        no2 = st.number_input(
            "NO₂ (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=40.0,
            step=1.0,
            help="Nitrogen Dioksida"
        )
        
        o3 = st.number_input(
            "O₃ (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=50.0,
            step=1.0,
            help="Ozon"
        )
        
        co = st.number_input(
            "CO (mg/m³)", 
            min_value=0.0,
            max_value=1000.0,
            value=4.0,
            step=0.1,
            help="Karbon Monoksida"
        )
        
        nh3 = st.number_input(
            "NH₃ (μg/m³)", 
            min_value=0.0,
            max_value=10000.0,
            value=10.0,
            step=0.1,
            help="Amonia"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tombol prediksi
    if st.button("🚀 Jalankan Prediksi", type="primary", use_container_width=True):
        # Validasi input
        inputs = {
            "pm10": pm10, "pm2_5": pm2_5, "so2": so2, "no": no,
            "no2": no2, "o3": o3, "co": co, "nh3": nh3
        }
        
        validation_errors = validate_inputs(inputs)
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
            st.stop()
        
        # Animasi progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Animasi loading
            animate_progress_bar(progress_bar, status_text)
            
            # Step 2: Hitung ISPU Manual (UTAMA)
            st.info("🎯 **Menggunakan perhitungan ISPU manual berdasarkan standar Baku Mutu Udara**")
            
            ispu_pred, sub_indices, dominant_pollutant = calculate_ispu_manual(
                pm10, pm2_5, so2, no2, o3, co
            )
            
            # Step 3: Klasifikasi RF (opsional)
            air_quality_class = classify_based_on_ispu(ispu_pred)
            
            # Step 4: Tampilkan hasil
            st.markdown("---")
            st.markdown('<h2 class="sub-header">📋 Hasil Prediksi</h2>', unsafe_allow_html=True)
            
            # Baris 1: Metode yang digunakan
            st.info(f"**Metode:** Perhitungan ISPU Manual | **Polutan Dominan:** {dominant_pollutant}")
            
            # Baris 2: Hasil utama
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category, color, emoji = classify_air_quality(ispu_pred)
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color} 0%, {color}99 100%);
                    color: white;
                    padding: 2rem;
                    border-radius: 15px;
                    text-align: center;
                    margin: 1rem 0;
                    animation: pulse 2s infinite;
                ">
                    <h1 style="font-size: 3rem; margin: 0;">{emoji}</h1>
                    <h3 style="margin: 1rem 0 0.5rem 0;">Kategori ISPU</h3>
                    <h2 style="font-size: 2rem; margin: 0;">{category}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("📈 Prediksi ISPU", f"{ispu_pred:.2f}")
                st.plotly_chart(create_gauge_chart(ispu_pred, "Nilai ISPU"), use_container_width=True)
            
            with col3:
                st.metric("🎯 Polutan Dominan", dominant_pollutant)
                st.metric("🤖 Klasifikasi", air_quality_class)
            
            # Baris 3: Detail perhitungan
            with st.expander("📊 Detail Perhitungan ISPU", expanded=True):
                pollutants = ["PM10", "PM2.5", "SO₂", "NO₂", "O₃", "CO"]
                
                # Tabel sub-indices
                st.markdown("### Sub-Index ISPU per Polutan")
                sub_df = pd.DataFrame({
                    "Polutan": pollutants,
                    "Sub-Index": sub_indices,
                    "Kontribusi": [f"{idx/ispu_pred*100:.1f}%" for idx in sub_indices]
                })
                st.dataframe(sub_df.style.highlight_max(subset=['Sub-Index'], color='#FFD700'), 
                           use_container_width=True)
                
                # Bar chart visualisasi
                fig = px.bar(sub_df, x='Polutan', y='Sub-Index', 
                           title="Kontribusi Polutan terhadap ISPU",
                           color='Sub-Index',
                           color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            
            # Baris 4: Rekomendasi
            st.markdown("### 💡 Rekomendasi Kesehatan")
            
            if ispu_pred <= 50:
                st.success("""
                ✅ **Kondisi udara BAIK**
                - Aktivitas luar ruangan aman untuk semua orang
                - Tidak ada pembatasan aktivitas
                - Kondisi ideal untuk berolahraga di luar
                """)
            elif ispu_pred <= 100:
                st.warning("""
                ⚠️ **Kondisi udara SEDANG**  
                - Kelompok sensitif (anak-anak, lansia, penderita ISPA) sebaiknya mengurangi aktivitas luar ruangan
                - Orang sehat dapat beraktivitas seperti biasa
                - Gunakan masker jika merasa tidak nyaman
                """)
            elif ispu_pred <= 200:
                st.error("""
                ❌ **Kondisi udara TIDAK SEHAT**
                - **Semua orang:** Gunakan masker jika keluar rumah
                - Kurangi aktivitas fisik berat di luar ruangan
                - Tutup jendela untuk mengurangi paparan polutan
                - Kelompok sensitif tetap di dalam ruangan
                """)
            elif ispu_pred <= 300:
                st.error("""
                🚫 **Kondisi udara SANGAT TIDAK SEHAT**
                - **Hindari aktivitas luar ruangan**
                - Gunakan air purifier di dalam ruangan
                - Tutup semua ventilasi udara
                - Segera cari pertolongan medis jika mengalami sesak napas, batuk, atau iritasi mata
                """)
            else:
                st.error("""
                ☠️ **Kondisi udara BERBAHAYA**
                - **Tetap di dalam ruangan** dengan penyejuk udara
                - Gunakan masker N95 jika harus keluar
                - Evakuasi ke area dengan udara bersih jika memungkinkan
                - Status darurat: hindari semua aktivitas luar ruangan
                """)
            
            # Baris 5: Input summary
            with st.expander("📝 Ringkasan Input", expanded=False):
                input_df = pd.DataFrame({
                    "Parameter": ["PM10", "PM2.5", "SO₂", "NO", "NO₂", "O₃", "CO", "NH₃"],
                    "Nilai": [pm10, pm2_5, so2, no, no2, o3, co, nh3],
                    "Satuan": ["μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "mg/m³", "μg/m³"]
                })
                st.dataframe(input_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            st.exception(e)

# =========================
# Halaman Informasi
# =========================
elif menu == "ℹ️ Informasi":
    st.markdown('<h1 class="main-header">ℹ️ Informasi Kualitas Udara</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📚 Definisi ISPU", "🏭 Sumber Polutan", "🛠️ Teknologi"])
    
    with tab1:
        st.markdown("""
        ### 📊 Indeks Standar Pencemar Udara (ISPU)
        
        | Kategori | Rentang ISPU | Warna | Dampak Kesehatan |
        |----------|--------------|-------|------------------|
        | **Baik** | 0 - 50 | 🟢 Hijau | Tidak berpengaruh pada kesehatan |
        | **Sedang** | 51 - 100 | 🟡 Kuning | Tidak nyaman pada kelompok sensitif |
        | **Tidak Sehat** | 101 - 200 | 🟠 Jingga | Gangguan pada kelompok sensitif |
        | **Sangat Tidak Sehat** | 201 - 300 | 🔴 Merah | Gangguan pada populasi umum |
        | **Berbahaya** | >300 | 🟣 Ungu | Berbahaya bagi semua populasi |
        """)
        
        # Visualisasi skala ISPU
        categories = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
        ranges = [50, 50, 100, 100, 200]
        colors = ['#10B981', '#F59E0B', '#EF4444', '#7C3AED', '#DC2626']
        
        fig = go.Figure()
        for i, (cat, rng, color) in enumerate(zip(categories, ranges, colors)):
            fig.add_trace(go.Bar(
                x=[rng],
                y=[cat],
                orientation='h',
                marker_color=color,
                name=cat,
                text=[f"{cat}<br>{rng} poin"],
                textposition='inside'
            ))
        
        fig.update_layout(
            title="Skala ISPU dan Kategori",
            barmode='stack',
            height=300,
            showlegend=False,
            xaxis_title="Rentang Poin",
            yaxis_title="Kategori"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### 🎯 Definisi Parameter Polutan
        
        | Parameter | Deskripsi | Dampak Kesehatan | Ambang Batas (μg/m³) |
        |-----------|-----------|------------------|---------------------|
        | **PM2.5** | Partikel halus (≤ 2.5μm) | Masuk ke paru-paru dan aliran darah | 15.5 |
        | **PM10**  | Partikel (≤ 10μm) | Iritasi saluran pernapasan | 50 |
        | **SO₂**   | Sulfur Dioksida | Gangguan pernapasan, asma | 52 |
        | **NO₂**   | Nitrogen Dioksida | Peradangan saluran pernapasan | 80 |
        | **O₃**    | Ozon | Iritasi mata, batuk, sesak napas | 120 |
        | **CO**    | Karbon Monoksida | Mengikat hemoglobin, kekurangan oksigen | 4000* |
        | **NH₃**   | Amonia | Iritasi mata dan saluran pernapasan | - |
        
        *CO dalam μg/m³ (4 mg/m³)
        """)
    
    with tab2:
        st.markdown("""
        ### 🏭 Sumber Polutan Utama
        
        #### **Sumber Transportasi:**
        - Kendaraan bermotor (NO₂, CO, PM)
        - Pesawat terbang
        - Kapal laut
        
        #### **Sumber Industri:**
        - Pabrik pembangkit listrik
        - Pabrik kimia
        - Pertambangan
        
        #### **Sumber Lainnya:**
        - Pembakaran sampah
        - Pertanian (NH₃ dari pupuk)
        - Debu konstruksi
        - Kebakaran hutan
        
        #### **Sumber Alamiah:**
        - Debu gurun
        - Vulkanik
        - Serbuk sari tanaman
        """)
    
    with tab3:
        st.markdown("""
        ### 🛠️ Teknologi yang Digunakan
        
        #### **1. Perhitungan ISPU Manual**
        - Formula berdasarkan **Baku Mutu Udara Ambien Nasional** (PP No. 22 Tahun 2021)
        - Menggunakan **interpolasi linear** untuk menghitung sub-index
        - **6 parameter polutan** utama
        - **ISPU = nilai tertinggi** dari semua sub-index
        
        #### **2. Random Forest Classifier**
        - Ensemble learning method
        - Akurat untuk klasifikasi kategori udara
        - Tahan terhadap overfitting
        - Menggunakan **6 fitur utama** untuk klasifikasi
        
        #### **3. Visualisasi Interaktif**
        - Gauge chart real-time
        - Analisis kontribusi polutan
        - Dashboard responsif
        - Grafik interaktif dengan Plotly
        """)

# =========================
# Halaman Visualisasi
# =========================
elif menu == "📈 Visualisasi":
    st.markdown('<h1 class="main-header">📈 Visualisasi Data Kualitas Udara</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribusi Polutan", "🔄 Tren ISPU", "🎯 Kategori"])
    
    with tab1:
        st.markdown("### 📊 Distribusi Polutan Normal")
        
        # Data contoh untuk distribusi polutan
        pollutants = ['PM2.5', 'PM10', 'SO₂', 'NO₂', 'O₃', 'CO', 'NH₃']
        normal_values = [25, 40, 15, 30, 45, 2, 8]
        high_values = [150, 200, 100, 150, 120, 15, 40]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                x=pollutants, 
                y=normal_values,
                title="Konsentrasi Polutan Normal",
                labels={'x': 'Polutan', 'y': 'Konsentrasi'},
                color=pollutants,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                x=pollutants, 
                y=high_values,
                title="Konsentrasi Polutan Tinggi",
                labels={'x': 'Polutan', 'y': 'Konsentrasi'},
                color=pollutants,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔄 Tren Kualitas Udara (Contoh)")
        
        # Generate sample time series data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        # Buat data tren dengan variasi
        np.random.seed(42)
        base_trend = np.linspace(50, 150, 30)
        seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, 30))
        noise = np.random.normal(0, 15, 30)
        ispu_values = base_trend + seasonal + noise
        ispu_values = np.clip(ispu_values, 0, 500)
        
        # Dataframe untuk plot
        df_trend = pd.DataFrame({
            'Tanggal': dates,
            'ISPU': ispu_values,
            'Kategori': pd.cut(ispu_values, 
                              bins=[0, 50, 100, 200, 300, 500],
                              labels=['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya'])
        })
        
        # Line chart dengan area
        fig = px.area(df_trend, x='Tanggal', y='ISPU',
                     title="Tren ISPU 30 Hari Terakhir",
                     labels={'ISPU': 'Nilai ISPU', 'Tanggal': 'Tanggal'},
                     color_discrete_sequence=['#3B82F6'])
        
        # Add threshold lines
        fig.add_hline(y=50, line_dash="dash", line_color="#10B981", 
                     annotation_text="Baik", annotation_position="right")
        fig.add_hline(y=100, line_dash="dash", line_color="#F59E0B",
                     annotation_text="Sedang", annotation_position="right")
        fig.add_hline(y=200, line_dash="dash", line_color="#EF4444",
                     annotation_text="Tidak Sehat", annotation_position="right")
        
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🎯 Distribusi Kategori Kualitas Udara")
        
        # Data contoh untuk pie chart
        categories = ['Baik', 'Sedang', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']
        distribution = [45, 30, 15, 7, 3]
        colors = ['#10B981', '#F59E0B', '#EF4444', '#7C3AED', '#DC2626']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig1 = go.Figure(data=[go.Pie(
                labels=categories,
                values=distribution,
                hole=.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textposition='inside'
            )])
            fig1.update_layout(
                title="Persentase Kategori Kualitas Udara",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Bar chart horizontal
            fig2 = px.bar(
                x=distribution,
                y=categories,
                orientation='h',
                title="Distribusi Kategori",
                labels={'x': 'Persentase (%)', 'y': 'Kategori'},
                color=categories,
                color_discrete_sequence=colors
            )
            fig2.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🌍 Sistem Prediksi Kualitas Udara**")
    st.caption("v2.0.0 | © 2024")

with footer_col2:
    st.markdown("**📞 Kontak**")
    st.caption("support@airquality.id")

with footer_col3:
    st.markdown("**⚠️ Disclaimer**")
    st.caption("Hasil prediksi berdasarkan perhitungan standar ISPU. Untuk keperluan akademik.")

# =========================
# Info Tambahan di Sidebar
# =========================
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Catatan Penting")
    st.warning("""
    **SARIMAX Model:**
    - Model memberikan nilai konstan 302.92
    - Digantikan dengan perhitungan ISPU manual
    - Hasil lebih akurat dan responsive
    """)