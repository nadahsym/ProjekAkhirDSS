import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score

# ============================================
# CONFIGURATION
# ============================================
DATA_FOLDER = Path("data")
MODEL_FOLDER = Path("model")
HISTORICAL_DATA = DATA_FOLDER / "historical_hr.csv"
MODEL_FILE = MODEL_FOLDER / "rf_model.pkl"
FEATURE_FILE = MODEL_FOLDER / "feature_list.json"

MODEL_FOLDER.mkdir(exist_ok=True)

# ============================================
# COLUMN NAME MAPPING (UNTUK FLEKSIBILITAS)
# ============================================
COLUMN_ALIASES = {
    'satisfaction_level': ['satisfaction', 'satisfaction_score', 'kepuasan', 'satisfaction_level'],
    'last_evaluation': ['evaluation', 'eval_score', 'last_eval', 'evaluation_score', 'last_evaluation'],
    'number_project': ['projects', 'num_projects', 'project_count', 'jumlah_proyek', 'number_project'],
    'average_montly_hours': ['monthly_hours', 'avg_hours', 'hours', 'jam_kerja', 'average_monthly_hours', 'average_montly_hours'],
    'time_spend_company': ['tenure', 'years', 'masa_kerja', 'time_spend_company'],
    'work_accident': ['accident', 'kecelakaan', 'work_accident'],
    'promotion_last_5years': ['promotion', 'promosi', 'promoted', 'promotion_last_5years'],
    'salary': ['gaji', 'salary', 'compensation'],
    'dept': ['department', 'departemen', 'divisi', 'dept'],
    'employee_id': ['id', 'emp_id', 'employee_id', 'karyawan_id'],
    'name': ['nama', 'employee_name', 'name'],
    'left': ['turnover', 'attrition', 'keluar', 'left']
}

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalisasi nama kolom berdasarkan COLUMN_ALIASES.
    Contoh: 'gaji' → 'salary', 'kepuasan' → 'satisfaction_level'
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    
    # Mapping
    column_mapping = {}
    for standard_name, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col in [a.lower() for a in aliases]:
                column_mapping[col] = standard_name
                break
    
    df = df.rename(columns=column_mapping)
    return df

# ============================================
# PREPROCESSING FUNCTION (USED FOR BOTH TRAIN & PREDICT)
# ============================================
def preprocess_data(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Konsisten preprocessing untuk training dan prediction.
    
    Steps:
    1. Remove duplicates
    2. Winsorization pada time_spend_company
    3. Encode salary (ordinal: low=1, medium=2, high=3)
    4. Simpan kolom 'dept' untuk output (tidak di-encode untuk fitur)
    5. Drop kolom yang tidak diperlukan
    6. Rename 'left' → 'turnover' jika training
    
    Args:
        df: Input dataframe
        is_training: True jika untuk training, False jika untuk prediksi
    
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    df = normalize_column_names(df)
    
    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Winsorization untuk time_spend_company
    if 'time_spend_company' in df.columns:
        Q1 = df['time_spend_company'].quantile(0.25)
        Q3 = df['time_spend_company'].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        df['time_spend_company_winsor'] = np.where(
            df['time_spend_company'] > upper, upper,
            np.where(df['time_spend_company'] < lower, lower, df['time_spend_company'])
        )
        df = df.drop('time_spend_company', axis=1)
    
    # 3. Encode salary (ordinal)
    if 'salary' in df.columns:
        salary_mapping = {"low": 1, "medium": 2, "high": 3}
        df['salary_encoded'] = df['salary'].map(salary_mapping)
        df = df.drop('salary', axis=1)
    
    # 4. Kolom 'dept' TIDAK di-encode, hanya disimpan untuk output
    # (tidak dipakai sebagai fitur prediksi)
    
    # 5. Rename target jika training
    if is_training and 'left' in df.columns:
        df = df.rename(columns={'left': 'turnover'})
    
    return df

# ============================================
# THRESHOLD OPTIMIZATION USING ROC CURVE
# ============================================
def calculate_optimal_threshold(y_true, y_pred_proba):
    """
    Hitung threshold optimal menggunakan Youden's Index dari ROC Curve.
    
    Args:
        y_true: True labels (0/1)
        y_pred_proba: Predicted probabilities
    
    Returns:
        optimal_threshold: Threshold optimal (float)
        roc_data: Dictionary dengan fpr, tpr, thresholds untuk visualisasi
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Youden's Index = TPR - FPR (maksimum gap antara TPR dan FPR)
    youdens_index = tpr - fpr
    optimal_idx = np.argmax(youdens_index)
    optimal_threshold = thresholds[optimal_idx]
    
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc_score(y_true, y_pred_proba),
        'optimal_idx': optimal_idx
    }
    
    return optimal_threshold, roc_data

# ============================================
# AUTO TRAINING FUNCTION (BACKGROUND)
# ============================================
def auto_train_model():
    """
    Fungsi training otomatis yang berjalan di background.
    Dipanggil saat model belum ada.
    Tidak menampilkan UI apapun ke HR.
    """
    # Load historical data
    if not HISTORICAL_DATA.exists():
        raise FileNotFoundError(
            f"Dataset historis tidak ditemukan: {HISTORICAL_DATA}\n"
            "Pastikan file 'historical_hr.csv' ada di folder 'data/'"
        )
    
    df = pd.read_csv(HISTORICAL_DATA)
    
    # Preprocessing
    df_clean = preprocess_data(df, is_training=True)
    
    # Pisahkan fitur dan target
    # Kolom yang TIDAK dipakai untuk prediksi
    exclude_cols = ['turnover', 'dept', 'employee_id', 'name']
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]
    
    X = df_clean[feature_cols]
    y = df_clean['turnover']
    
    # Train model dengan parameter yang diminta
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save model dan feature list
    joblib.dump(model, MODEL_FILE)
    with open(FEATURE_FILE, 'w') as f:
        json.dump(feature_cols, f)
    
    return model, feature_cols

# ============================================
# LOAD OR TRAIN MODEL
# ============================================
def get_model():
    """
    Load model jika sudah ada, atau train otomatis jika belum.
    Proses ini tidak tampil di UI HR.
    """
    if MODEL_FILE.exists() and FEATURE_FILE.exists():
        # Load existing model
        model = joblib.load(MODEL_FILE)
        with open(FEATURE_FILE, 'r') as f:
            features = json.load(f)
        return model, features
    else:
        # Auto train
        return auto_train_model()

# ============================================
# ALIGN FEATURES (UNTUK DATASET BARU)
# ============================================
def align_features(df: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """
    Pastikan dataset baru memiliki kolom yang sama dengan training.
    - Kolom hilang → dibuat dan diisi 0
    - Kolom tidak diperlukan → dihapus
    - Urutan kolom disamakan
    """
    df = df.copy()
    
    # Tambahkan kolom yang hilang
    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # Hapus kolom yang tidak diperlukan
    df = df[required_features]
    
    return df

# ============================================
# SHAP ANALYSIS (LOCAL FEATURE IMPORTANCE) WITH SMART SAMPLING
# ============================================
def calculate_shap_importance(model, X_data, feature_names, max_samples=1000):
    """
    Hitung SHAP values untuk dataset yang diupload dengan strategi sampling cerdas.
    Menggunakan TreeExplainer untuk Random Forest.
    
    Strategy:
    - Jika dataset ≤ max_samples: hitung semua
    - Jika dataset > max_samples: sampling stratified berdasarkan predicted probability
    
    Args:
        model: Trained model
        X_data: Feature data (DataFrame atau array)
        feature_names: List nama fitur
        max_samples: Maximum samples untuk SHAP calculation (default: 1000)
    
    Returns:
        shap_importance_df: DataFrame dengan feature dan mean_abs_shap
        shap_values: SHAP values untuk visualisasi
        sample_info: Dictionary dengan info sampling
    """
    n_samples = len(X_data)
    sample_info = {
        'total_data': n_samples,
        'sampled': False,
        'sample_size': n_samples
    }
    
    # Tentukan apakah perlu sampling
    if n_samples > max_samples:
        
        # Predict probability untuk stratification
        y_pred_proba = model.predict_proba(X_data)[:, 1]
        
        # Buat bins untuk stratifikasi (low, medium, high risk)
        bins = [0, 0.3, 0.5, 1.0]
        labels = ['low', 'medium', 'high']
        risk_categories = pd.cut(y_pred_proba, bins=bins, labels=labels)
        
        # Hitung proporsi sampling per kategori
        sample_indices = []
        for category in labels:
            cat_indices = np.where(risk_categories == category)[0]
            if len(cat_indices) > 0:
                # Proporsi sampling berdasarkan ukuran kategori
                n_category_samples = min(
                    len(cat_indices),
                    int(max_samples * len(cat_indices) / n_samples)
                )
                # Minimum 100 per kategori jika memungkinkan
                n_category_samples = max(n_category_samples, min(100, len(cat_indices)))
                
                sampled_cat = np.random.choice(
                    cat_indices,
                    size=n_category_samples,
                    replace=False
                )
                sample_indices.extend(sampled_cat)
        
        sample_indices = np.array(sample_indices)
        X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
        
        sample_info['sampled'] = True
        sample_info['sample_size'] = len(sample_indices)
        sample_info['sampling_method'] = 'stratified_by_risk'
        
    else:
        X_sample = X_data
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)

    # Jika binary classification, ambil class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Jika SHAP menghasilkan array 3D (n_samples, n_features, 1)
    # ubah menjadi 2D (n_samples, n_features)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    
    # Hitung mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'Pengaruh Terhadap Risiko Turnover': mean_abs_shap
    }).sort_values('Pengaruh Terhadap Risiko Turnover', ascending=False).reset_index(drop=True)
    
    return shap_importance_df, shap_values, sample_info

# What-If Simulation Module
def run_intervention_simulation(df_features, model, interventions, feature_list, high_thresh, medium_thresh):
    """
    df_features : pd.DataFrame aligned dengan feature_list (numeric encoded sesuai model)
    model : trained classifier with predict_proba
    interventions : list of dicts, each {'feature': str, 'type': 'add'|'set'|'scale', 'value': float}
    feature_list : list of features in the same order as model expects (columns present in df_features)
    returns dict with old/new metrics and per-row delta
    """
    # Keep original predictions
    old_proba = model.predict_proba(df_features[feature_list])[:, 1] * 100  # percent

    # Create simulated copy
    df_sim = df_features.copy()

    # Apply interventions one by one
    for it in interventions:
        f = it['feature']
        typ = it.get('type', 'add')
        val = it.get('value', 0.0)
        if f not in df_sim.columns:
            # ignore or warn silently
            continue

        # handle numeric (most cases) - safer to clip if bounded (example: satisfaction 0..1)
        if typ == 'add':
            df_sim[f] = df_sim[f] + val
        elif typ == 'set':
            df_sim[f] = val
        elif typ == 'scale':
            df_sim[f] = df_sim[f] * val
        # optional clipping for common ranges
        if f in ['satisfaction_level', 'last_evaluation']:  # add other bounded features if any
            df_sim[f] = df_sim[f].clip(0, 1)
        # if categorical encoded (e.g. salary_encoded) and intervention is set to specific category index,
        # user must ensure df_sim has the right representation (advanced)

    # Predict new probabilities
    new_proba = model.predict_proba(df_sim[feature_list])[:, 1] * 100

    # Metrics
    old_avg = old_proba.mean()
    new_avg = new_proba.mean()
    avg_drop = old_avg - new_avg

    old_high = int((old_proba >= high_thresh).sum())
    new_high = int((new_proba >= high_thresh).sum())
    old_medium = int(((old_proba >= medium_thresh) & (old_proba < high_thresh)).sum())
    new_medium = int(((new_proba >= medium_thresh) & (new_proba < high_thresh)).sum())
    old_low = int((old_proba < medium_thresh).sum())
    new_low = int((new_proba < medium_thresh).sum())

    # Per-row delta (optional)
    delta_perc = new_proba - old_proba

    return {
        'old_avg': old_avg, 'new_avg': new_avg, 'avg_drop': avg_drop,
        'old_counts': {'high': old_high, 'medium': old_medium, 'low': old_low},
        'new_counts': {'high': new_high, 'medium': new_medium, 'low': new_low},
        'delta_perc': delta_perc,
        'old_proba': old_proba,
        'new_proba': new_proba
    }

# ============================================
# STREAMLIT UI (FOR HR ONLY)
# ============================================
st.set_page_config(
    page_title="DSS - Employee Turnover Prediction",
    page_icon="🏢",
    layout="wide"
)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
page = st.sidebar.selectbox(
    "📍 Navigation",
    ["🏠 Home", "📊 Turnover Prediction", "📈 Organizational Insights"]
)

# Load model (otomatis train jika belum ada)
try:
    with st.spinner("🔄 Initializing system..."):
        model, feature_list = get_model()
    st.success("✅ System ready!")
except Exception as e:
    st.error(f"❌ Error initializing system: {str(e)}")
    st.stop()

    st.title("🏠 HR Dashboard - Employee Turnover DSS")
    st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ SHAP Settings")
max_shap_samples = st.sidebar.slider(
    "Max samples untuk SHAP",
    min_value=500,
    max_value=5000,
    value=1000,
    step=100,
    help="Jumlah sampel maksimum untuk kalkulasi SHAP. Lebih banyak = lebih akurat tapi lebih lama."
)
st.session_state['max_shap_samples'] = max_shap_samples

# SIDEBAR THRESHOLD SETTINGS
with st.sidebar:
    st.markdown("### 🎯 Risk Threshold Settings")

    # Load default jika belum ada
    if 'high_threshold' not in st.session_state:
        st.session_state['high_threshold'] = 50.0
    if 'medium_threshold' not in st.session_state:
        st.session_state['medium_threshold'] = 30.0

    # Cek apakah threshold optimal bisa dihitung dari historical data
    optimal_threshold = None
    if HISTORICAL_DATA.exists():
        try:
            df_hist = pd.read_csv(HISTORICAL_DATA)
            df_hist_clean = preprocess_data(df_hist, is_training=True)

            exclude_cols = ['turnover', 'dept', 'employee_id', 'name']
            feature_cols = [c for c in df_hist_clean.columns if c not in exclude_cols]

            X_hist = df_hist_clean[feature_cols]
            X_hist_aligned = align_features(X_hist, feature_list)
            y_hist = df_hist_clean['turnover']
            y_pred_proba = model.predict_proba(X_hist_aligned)[:, 1]

            optimal_threshold, roc_data = calculate_optimal_threshold(y_hist, y_pred_proba)
        except:
            pass

    # PILIH MODE
    threshold_mode = st.radio(
        "Mode Pengaturan Threshold",
        ["Optimal (Recommended)", "Manual"],
        help="Gunakan optimal jika ingin threshold berdasarkan ROC curve"
    )

    # KALKULASI THRESHOLD
    if threshold_mode == "Optimal (Recommended)" and optimal_threshold is not None:
        high_threshold = optimal_threshold * 100
        st.info(f"Threshold optimal berdasarkan ROC: **{high_threshold:.1f}%**")
    else:
        high_threshold = st.slider(
            "High Risk Threshold (%)",
            min_value=30.0,
            max_value=70.0,
            value=st.session_state['high_threshold'],
            step=1.0,
            help="Karyawan dengan probabilitas ≥ nilai ini dianggap High Risk"
        )

    medium_threshold = round(high_threshold * 0.6, 1)

    # Simpan ke session state
    st.session_state['high_threshold'] = high_threshold
    st.session_state['medium_threshold'] = medium_threshold

    # Tampilkan kategori risiko
    st.markdown("### 📊 Risk Category")
    st.write(f"🔴 **High Risk:** ≥ {high_threshold:.1f}%")
    st.write(f"🟡 **Medium Risk:** {medium_threshold:.1f}% – {high_threshold:.1f}%")
    st.write(f"🟢 **Low Risk:** < {medium_threshold:.1f}%")


# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    st.title("🏢 Decision Support System")
    st.subheader("Employee Turnover Prediction & Risk Analysis")
    st.markdown("---")

    st.markdown("""
    ### 👋 Selamat Datang di Dashboard DSS HR
    Sistem ini membantu HR dalam:
    - 🔮 **Memprediksi risiko turnover karyawan**
    - 🎯 **Mengidentifikasi faktor yang paling berpengaruh**
    - 💡 **Memberikan rekomendasi keputusan berbasis data**
    - 🧠 **Mendukung program retensi & engagement**
    ---

    ### 📌 Sebelum mulai
    Silakan atur:
    - **Risk Threshold Settings** di panel **sidebar** untuk menentukan batas kategori risiko (High / Medium / Low)
    - **SHAP Settings** untuk mengatur jumlah sampel analisis faktor penyebab turnover

    Setelah itu, lanjutkan ke tab **📊 Turnover Prediction** untuk upload dataset karyawan dan melihat hasil prediksi.

    ---
    """)
    
    # Insight placeholder
    if 'predictions' in st.session_state:
        df = st.session_state['predictions']
        st.markdown("### 📍 Ringkasan Terbaru")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Karyawan", len(df))
        with col2:
            st.metric("High Risk", len(df[df['turnover_probability (%)']>=50]))
        with col3:
            st.metric("Rata-rata Risiko", f"{df['turnover_probability (%)'].mean():.2f}%")

    st.stop()

# ============================================
# MAIN UI: UPLOAD & PREDICT
# ============================================
elif page == "📊 Turnover Prediction":

    st.title("📊 Turnover Prediction")
    st.markdown("Prediksi risiko turnover karyawan secara individual berdasarkan data historis, "
            "dengan identifikasi kategori risiko dan faktor penyebab utama melalui analisis SHAP.")
    st.markdown("---")

    st.markdown("### 📤 Upload Dataset Karyawan")
    st.info("Upload file CSV yang berisi data karyawan yang ingin diprediksi. **Tidak perlu** menyertakan kolom 'left' atau 'Turnover'.")
    # GUIDE KOLOM YANG DIBUTUHKAN
    with st.expander("📋 Lihat Kolom yang Dibutuhkan & Contoh Format"):
        st.markdown("""
        **Kolom Wajib untuk Prediksi:**
        
        | Kolom | Alias yang Diterima | Tipe Data | Contoh | Keterangan |
        |-------|-------------------|-----------|---------|------------|
        | `satisfaction_level` | satisfaction, kepuasan | Float (0-1) | 0.78 | Tingkat kepuasan karyawan |
        | `last_evaluation` | evaluation, eval_score | Float (0-1) | 0.85 | Skor evaluasi terakhir |
        | `number_project` | projects, jumlah_proyek | Integer | 5 | Jumlah proyek yang ditangani |
        | `average_montly_hours` | monthly_hours, jam_kerja | Integer | 180 | Rata-rata jam kerja per bulan |
        | `time_spend_company` | tenure, masa_kerja | Integer | 3 | Lama bekerja (tahun) |
        | `work_accident` | accident, kecelakaan | Binary (0/1) | 0 | Pernah kecelakaan kerja |
        | `promotion_last_5years` | promotion, promosi | Binary (0/1) | 0 | Promosi dalam 5 tahun terakhir |
        | `salary` | gaji, compensation | Categorical | low/medium/high | Tingkat gaji |
        
        **Kolom Opsional (untuk identifikasi):**
        - `employee_id` / `id` / `emp_id`
        - `name` / `nama` / `employee_name`
        - `dept` / `department` / `departemen`
        
        **💡 Tips:**
        - Sistem akan otomatis mengenali nama kolom meskipun tidak persis sama (misalnya 'gaji' akan dikenali sebagai 'salary')
        - Jika kolom tidak ada, sistem akan mengisi dengan nilai default (0)
        """)

    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=['csv'],
        help="Sistem mendukung berbagai nama kolom (misal: 'gaji' = 'salary', 'kepuasan' = 'satisfaction_level')"
    )

    if uploaded_file is not None:
        try:
            # Load data
            df_new = pd.read_csv(uploaded_file)
            
            st.markdown("### 📊 Preview Data")
            st.dataframe(df_new.head(10), use_container_width=True)
            st.caption(f"Total karyawan: **{len(df_new)}**")
            
            # Simpan kolom untuk output (employee_id, name, dept)
            output_cols = {}
            if 'employee_id' in df_new.columns:
                output_cols['employee_id'] = df_new['employee_id']
            if 'name' in df_new.columns:
                output_cols['name'] = df_new['name']
            if 'dept' in df_new.columns:
                output_cols['dept'] = df_new['dept']
            
            # Button: Predict
            if st.button("🔮 Jalankan Prediksi", type="primary", use_container_width=True):
                with st.spinner("⚙️ Processing data dan melakukan prediksi..."):
                    # Preprocessing
                    df_processed = preprocess_data(df_new, is_training=False)
                    
                    # Align features
                    df_features = align_features(df_processed, feature_list)
                    
                    # Predict
                    predictions = model.predict_proba(df_features)[:, 1] * 100  # Probability dalam %
                    
                    st.session_state['model'] = model
                    st.session_state['feature_list'] = feature_list
                    st.session_state['df_features'] = df_features
                    st.session_state['predictions_raw'] = predictions
                    st.session_state['original_processed'] = df_processed

                    # Prepare output
                    result_df = pd.DataFrame()
                    for col_name, col_data in output_cols.items():
                        result_df[col_name] = col_data.values
                    result_df['turnover_probability (%)'] = predictions.round(2)
                    
                    # Sort by risk (highest first)
                    result_df = result_df.sort_values('turnover_probability (%)', ascending=False).reset_index(drop=True)
                    
                    # Calculate SHAP importance untuk dataset UPLOAD (bukan training)
                    max_samples = st.session_state.get('max_shap_samples', 1000)
                    
                    shap_importance_df, shap_values, sample_info = calculate_shap_importance(
                        model, df_features, feature_list, max_samples=max_samples
                    )
                    
                    # Store in session
                    st.session_state['predictions'] = result_df
                    st.session_state['feature_importance'] = shap_importance_df  
                    st.session_state['shap_values'] = shap_values
                    st.session_state['shap_sample_info'] = sample_info 
                    st.session_state['df_features'] = df_features
                    st.session_state['model'] = model
                    st.session_state['original_df'] = df_new

                
                st.success("✅ Prediksi selesai!")
                st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error saat memproses file: {str(e)}")
            st.stop()

    # ============================================
    # DISPLAY PREDICTIONS
    # ============================================
    if 'predictions' in st.session_state:
        st.markdown("---")
        st.markdown("## 🎯 Hasil Prediksi")
        
        predictions_df = st.session_state['predictions']
        
        # Metrics (menggunakan threshold dari session state jika ada)
        high_thresh = st.session_state.get('high_threshold', 50.0)
        medium_thresh = st.session_state.get('medium_threshold', 30.0)

        col1, col2, col3 = st.columns(3)
        with col1:
            high_risk = len(predictions_df[predictions_df['turnover_probability (%)'] >= high_thresh])
            st.metric(f"🔴 High Risk (≥{high_thresh:.1f}%)", high_risk)
        with col2:
            medium_risk = len(predictions_df[(predictions_df['turnover_probability (%)'] >= medium_thresh) & 
                                            (predictions_df['turnover_probability (%)'] < high_thresh)])
            st.metric(f"🟡 Medium Risk ({medium_thresh:.1f}-{high_thresh:.1f}%)", medium_risk)
        with col3:
            low_risk = len(predictions_df[predictions_df['turnover_probability (%)'] < medium_thresh])
            st.metric(f"🟢 Low Risk (<{medium_thresh:.1f}%)", low_risk)
        
        st.markdown("### 📋 Ranking Karyawan Berdasarkan Risiko")
        st.dataframe(
            predictions_df.style.background_gradient(
                subset=['turnover_probability (%)'],
                cmap='RdYlGn_r',
                vmin=0,
                vmax=100
            ),
            use_container_width=True,
            height=400
        )
        
        # ============================================
        # FEATURE IMPORTANCE
        # ============================================
        st.markdown("---")
        st.markdown("## 📊 Faktor-Faktor Paling Berpengaruh (SHAP Analysis)")
        
        # Info sampling
        sample_info = st.session_state.get('shap_sample_info', {})
        if sample_info.get('sampled', False):
            st.info(
                f"💡 Analisis SHAP menggunakan **{sample_info['sample_size']:,} sampel** dari **{sample_info['total_data']:,} karyawan** "
                f"(stratified sampling berdasarkan tingkat risiko). Hasil tetap akurat dan representatif."
            )
        else:
            st.info(f"💡 Analisis ini berdasarkan **{sample_info.get('total_data', 'semua')} data** yang Anda upload.")
        
        feature_imp = st.session_state['feature_importance']
        shap_values = st.session_state['shap_values']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Top 10 Features (SHAP)")
            # Tabel SHAP
            st.dataframe(
                feature_imp.head(10).style.format({'Pengaruh Terhadap Risiko Turnover': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
        
        with col2:
            st.markdown("### Top 3 Faktor Utama")
            top3 = feature_imp.head(3)
            for idx, row in top3.iterrows():
                st.metric(
                    label=row['feature'],
                    value=f"{row['Pengaruh Terhadap Risiko Turnover']:.4f}",
                    help="Mean absolute SHAP value - semakin tinggi, semakin berpengaruh"
                )
        
        # ============================================
        # RECOMMENDATIONS (berdasarkan Top 3 Feature Importance)
        # ============================================
        st.markdown("---")
        st.markdown("## 💡 Rekomendasi Tindakan Berdasarkan Faktor Utama")

        top3 = feature_imp.head(3)['feature'].tolist()

        # Mapping rekomendasi berbasis feature
        recommendation_map = {
            "satisfaction_level": {
                "title": "Program Peningkatan Kepuasan Karyawan",
                "action": "Adakan sesi feedback rutin, perbaiki workload, dan perkuat hubungan atasan-bawahan.",
                "impact": "Signifikan pada retensi & komitmen karyawan."
            },
            "average_montly_hours": {
                "title": "Manajemen Beban Kerja & Work-life Balance",
                "action": "Atur distribusi jam kerja yang sehat dan pastikan waktu istirahat terkontrol.",
                "impact": "Menurunkan kelelahan, burnout, dan niat resign."
            },
            "time_spend_company": {
                "title": "Career Path & Recognition",
                "action": "Berikan peluang rotasi, training skill, dan pengakuan formal atas kinerja.",
                "impact": "Meningkatkan loyalitas karyawan senior maupun lama bekerja."
            },
            "number_project": {
                "title": "Optimasi Alokasi Tugas",
                "action": "Pastikan jumlah project sesuai kapasitas dan lakukan review beban kerja mingguan.",
                "impact": "Mengurangi stres berlebih dan meningkatkan engagement."
            },
            "promotion_last_5years": {
                "title": "Program Keadilan Karir & Promosi",
                "action": "Evaluasi fairness promosi dan transparansi penilaian kinerja.",
                "impact": "Meningkatkan rasa keadilan & motivasi kerja."
            },
            "salary_encoded": {
                "title": "Kompensasi & Benefit Kompetitif",
                "action": "Lakukan benchmarking gaji & benefit, serta implementasi reward berbasis kinerja.",
                "impact": "Mengurangi turnover karena ketidakpuasan kompensasi."
            },
            "Work_accident": {
                "title": "Keselamatan & Kesehatan Kerja",
                "action": "Perkuat standar K3 dan pelatihan keamanan berdasarkan insiden terakhir.",
                "impact": "Meningkatkan keamanan & kepercayaan kepada perusahaan."
            }
        }

        # Tampilkan rekomendasi berdasarkan top 3 fitur:
        for i, feature in enumerate(top3, 1):
            rec = recommendation_map.get(feature, None)
            with st.expander(f"**{i}. {rec['title'] if rec else feature.replace('_',' ').title()}**"):
                if rec:
                    st.write(f"**Faktor penyebab utama:** `{feature}`")
                    st.write(f"**Rekomendasi tindakan:** {rec['action']}")
                    st.write(f"**Manfaat / Dampak Bisnis:** {rec['impact']}")
                else:
                    st.info("Belum tersedia rekomendasi spesifik untuk faktor ini — dapat ditentukan berdasarkan analisis HR lebih lanjut.")

        
        # ============================================
        # DOWNLOAD RESULTS
        # ============================================
        st.markdown("---")
        st.markdown("## 📥 Download Hasil Prediksi")
        
        csv = predictions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download predictions.csv",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# ==========================
# TAB 3 - ORGANIZATIONAL INSIGHTS
# ==========================
elif page == "📈 Organizational Insights":

    st.title("📈 Organizational Insights")
    st.markdown("Analisis risiko turnover tingkat organisasi & simulasi dampak strategi intervensi.")
    st.markdown("---")

    # Pastikan prediksi sudah ada
    if 'predictions' not in st.session_state:
        st.warning("⚠️ Tolong jalankan prediksi terlebih dahulu di tab Prediction & Analysis.")
        st.stop()

    predictions_df = st.session_state['predictions']
    feature_imp = st.session_state['feature_importance']

    # ======== Department Comparison ========
    st.markdown("## 🧩 Perbandingan Risiko Turnover per Departemen")

    if 'dept' in predictions_df.columns:
        dept_risk = predictions_df.groupby('dept')['turnover_probability (%)'].mean().reset_index()
        dept_risk = dept_risk.sort_values('turnover_probability (%)', ascending=False)

        st.dataframe(
            dept_risk.style.background_gradient(
                subset=['turnover_probability (%)'],
                cmap='RdYlGn_r'
            ),
            use_container_width=True
        )

        highest = dept_risk.iloc[0]
        st.info(
            f"📍 **Prioritas utama:** Departemen **{highest['dept']}** memiliki risiko turnover rata-rata tertinggi yaitu **{highest['turnover_probability (%)']:.2f}%**."
        )

    else:
        st.error("Dataset tidak memiliki kolom departemen.")
    
    st.markdown("---")

    # ======== Simulasi Intervensi Strategi ========
    st.markdown("## 🔬 Simulasi Intervensi Strategi (What-If) - Level Organisasi")
    st.write("Simulasi ini menghitung ulang prediksi model jika kita mengubah satu atau beberapa faktor penting bagi seluruh karyawan.")

    # Pastikan prerequisites: model, feature list, & df_features tersedia
    if 'model' not in st.session_state:
        st.warning("⚠ Model belum tersedia. Silakan jalankan prediksi terlebih dahulu pada halaman '📊 Turnover Prediction'.")
        st.stop()

    if 'feature_list' not in st.session_state:
        st.warning("⚠ Daftar fitur belum dimuat. Silakan hilangkan dulu prediksi.")
        st.stop()

    if 'df_features' not in st.session_state:
        st.warning("⚠ Data fitur untuk simulasi belum tersedia. Silakan jalankan prediksi terlebih dahulu.")
        st.stop()

    # Load variables from session_state
    model = st.session_state['model']
    feature_list = st.session_state['feature_list']
    df_feat_source = st.session_state['df_features'].copy()

    if df_feat_source is not None:
        # show top features to choose (jika ada feature_importance)
        feature_imp = st.session_state.get('feature_importance', None)
        if feature_imp is not None:
            top_features = feature_imp.head(5)['feature'].tolist()
        else:
            top_features = [c for c in df_feat_source.columns if c in feature_list][:5]

        st.markdown("### Pilih skenario simulasi cepat (Top features)")
        col_a, col_b = st.columns(2)
        with col_a:
            sel_top = st.multiselect("Pilih fitur (Top suggestions):", top_features, default=top_features[:1])
        with col_b:
            manual_feature = st.text_input("Atau masukkan nama fitur manual (kosong = skip):", "")

        # Build interventions list UI
        interventions = []
        chosen = sel_top.copy()
        if manual_feature.strip():
            chosen.append(manual_feature.strip())

        st.markdown("### Atur perubahan yang akan diterapkan ke fitur terpilih")
        for f in chosen:
            st.markdown(f"**{f}**")
            typ = st.selectbox(f"Tipe perubahan untuk {f}", options=['add','set','scale'], key=f+"_typ")
            if typ == 'add':
                val = st.number_input(f"Tambahkan nilai ke {f} (mis. +0.10):", value=0.10, step=0.01, key=f+"_val")
            elif typ == 'set':
                val = st.number_input(f"Set nilai {f} menjadi (mis. 1.0):", value=1.0, step=0.01, key=f+"_val")
            else:
                val = st.number_input(f"Kalikan {f} dengan (mis. 0.9 untuk reduce):", value=0.9, step=0.01, key=f+"_val")
            interventions.append({'feature': f, 'type': typ, 'value': float(val)})

        # Run
        if st.button("Jalankan Simulasi"):
            # thresholds dari session
            high_thresh = st.session_state.get('high_threshold', 50.0)
            medium_thresh = st.session_state.get('medium_threshold', 30.0)

            with st.spinner("Menghitung ulang prediksi..."):
                try:
                    result = run_intervention_simulation(df_feat_source, model, interventions, feature_list, high_thresh, medium_thresh)
                except Exception as e:
                    st.error(f"Gagal menjalankan simulasi: {e}")
                    result = None

            if result:
                # Ringkasan numerik
                st.markdown("### Hasil Ringkasan")
                st.write(f"- Rata-rata probabilitas (sebelum): **{result['old_avg']:.2f}%**")
                st.write(f"- Rata-rata probabilitas (sesudah): **{result['new_avg']:.2f}%**")
                st.write(f"- Penurunan rata-rata: **{result['avg_drop']:.2f} poin persentase**")
                st.write("**Perubahan jumlah berdasarkan kategori (High/Medium/Low):**")
                st.write(f"- High : {result['old_counts']['high']} → {result['new_counts']['high']}")
                st.write(f"- Medium : {result['old_counts']['medium']} → {result['new_counts']['medium']}")
                st.write(f"- Low : {result['old_counts']['low']} → {result['new_counts']['low']}")

                # Optional: show top N employees with largest drop/increase
                deltas = pd.Series(result['delta_perc'])
                top_drop_idx = deltas.nlargest(5).index  # biggest increase in prob (negative = improvement?), careful sign
                st.markdown("#### Contoh karyawan terpengaruh paling besar (perubahan probabilitas)")
                sample_table = pd.DataFrame({
                    'employee_index': top_drop_idx,
                    'old_proba': result['old_proba'][top_drop_idx],
                    'new_proba': result['new_proba'][top_drop_idx],
                    'delta_pct': result['delta_perc'][top_drop_idx]
                })
                st.dataframe(sample_table.style.format({'old_proba':'{:.2f}','new_proba':'{:.2f}','delta_pct':'{:.2f}'}), use_container_width=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🏢 Decision Support System for HR | Developed by IT/Data Team")