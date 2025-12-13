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
# CUSTOM CSS FOR BETTER UI
# ============================================
def load_custom_css():
    st.markdown("""
    <style>
        /* Main container padding */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Card-like containers */
        .stExpander {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 10px;
            border-left: 4px solid;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* DataFrame styling */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Section headers */
        h2 {
            color: #1f77b4;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 1.5rem;
        }
        
        h3 {
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        /* Icon styling */
        .icon-header {
            font-size: 1.2em;
            margin-right: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# STREAMLIT UI (FOR HR ONLY)
# ============================================
st.set_page_config(
    page_title="Employee Turnover Prediction System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=HR+Analytics", use_container_width=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

page = st.sidebar.radio(
    "📍 **Navigation**",
    ["🏠 Home", "📊 Turnover Prediction", "📈 Organizational Insights"],
    label_visibility="visible"
)

# Load model (otomatis train jika belum ada)
try:
    with st.spinner("🔄 Initializing system..."):
        model, feature_list = get_model()
    st.sidebar.success("✅ System Ready")
except Exception as e:
    st.error(f"❌ **Error initializing system:** {str(e)}")
    st.stop()

st.sidebar.markdown("---")

# ============================================
# SIDEBAR SETTINGS
# ============================================
with st.sidebar:
    st.markdown("### ⚙️ **Analysis Settings**")
    
    with st.expander("🎯 **Risk Thresholds**", expanded=False):
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
            "**Threshold Mode**",
            ["🎓 Optimal (ROC-based)", "✋ Manual"],
            help="Optimal mode uses statistical analysis to determine best thresholds"
        )

        # KALKULASI THRESHOLD
        if threshold_mode == "🎓 Optimal (ROC-based)" and optimal_threshold is not None:
            high_threshold = optimal_threshold * 100
            st.success(f"📊 Optimal threshold: **{high_threshold:.1f}%**")
        else:
            high_threshold = st.slider(
                "High Risk Threshold (%)",
                min_value=30.0,
                max_value=70.0,
                value=st.session_state['high_threshold'],
                step=1.0,
                help="Employees with probability ≥ this value are considered High Risk"
            )

        medium_threshold = round(high_threshold * 0.6, 1)

        # Simpan ke session state
        st.session_state['high_threshold'] = high_threshold
        st.session_state['medium_threshold'] = medium_threshold

        # Tampilkan kategori risiko dengan visual yang lebih baik
        st.markdown("**Risk Categories:**")
        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; margin: 5px 0;'>
            🔴 <strong>High Risk:</strong> ≥ {high_threshold:.1f}%<br>
            🟡 <strong>Medium Risk:</strong> {medium_threshold:.1f}% - {high_threshold:.1f}%<br>
            🟢 <strong>Low Risk:</strong> < {medium_threshold:.1f}%
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🧠 **SHAP Analysis**", expanded=False):
        max_shap_samples = st.slider(
            "Max samples for SHAP calculation",
            min_value=500,
            max_value=5000,
            value=1000,
            step=100,
            help="Higher = more accurate but slower processing"
        )
        st.session_state['max_shap_samples'] = max_shap_samples

st.sidebar.markdown("---")
st.sidebar.caption("💼 HR Analytics System v2.0")

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;'>
            🏢 Employee Turnover Prediction System
        </h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: 0;'>
            AI-Powered Decision Support for Strategic HR Management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white; 
                    height: 250px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center;'>
            <h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>🔮</h2>
            <h4 style='color: white; margin: 0.5rem 0;'>Predict</h4>
            <p style='margin: 0; font-size: 0.9rem;'>Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;
                    height: 250px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center;'>
            <h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>🎯</h2>
            <h4 style='color: white; margin: 0.5rem 0;'>Identify</h4>
            <p style='margin: 0; font-size: 0.9rem;'>Key Factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;
                    height: 250px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; '>
            <h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>💡</h2>
            <h4 style='color: white; margin: 0.5rem 0; font-size: 1.25rem; line-height: 1.1;'>Recommend</h4>
            <p style='margin: 0; font-size: 0.9rem;'>Action Plans</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; color: white;
                    height: 250px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center;'>
            <h2 style='color: white; border: none; margin: 0; font-size: 2.5rem;'>🧠</h2>
            <h4 style='color: white; margin: 0.5rem 0; '>Support</h4>
            <p style='margin: 0; font-size: 0.9rem;'>Retention Strategy</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Getting Started Section
    st.markdown("## 🚀 Getting Started")
    
    tab1, tab2, tab3 = st.tabs(["📋 Quick Start", "⚙️ Configuration", "📊 Dashboard Overview"])
    
    with tab1:
        st.markdown("""
        ### Welcome to Your HR Analytics Dashboard
        
        This system helps you make data-driven decisions about employee retention using advanced machine learning.
        
        #### 📝 Step-by-Step Guide:
        
        1. **Configure Settings** (Sidebar)
           - Set risk thresholds (Optimal or Manual mode)
           - Adjust SHAP analysis parameters
        
        2. **Upload Employee Data** (Turnover Prediction tab)
           - Prepare your CSV file with employee information
           - System auto-recognizes various column names
        
        3. **Review Predictions**
           - See risk levels for each employee
           - Identify high-risk individuals
        
        4. **Analyze Insights**
           - Understand key factors driving turnover
           - Get actionable recommendations
        
        5. **Simulate Interventions** (Organizational Insights tab)
           - Test "what-if" scenarios
           - Plan strategic interventions
        """)
    
    with tab2:
        st.markdown("""
        ### ⚙️ System Configuration
        
        **Risk Threshold Settings**
        - Configure in the sidebar under "Risk Thresholds"
        - Choose between Optimal (ROC-based) or Manual mode
        - Define your risk categories: High, Medium, Low
        
        **SHAP Analysis Settings**
        - Adjust sample size for factor analysis
        - Balance between accuracy and processing speed
        - Recommended: 1000 samples for datasets < 10,000 rows
        
        **Current Settings:**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**High Risk Threshold:** {st.session_state.get('high_threshold', 50):.1f}%")
            st.info(f"**Medium Risk Threshold:** {st.session_state.get('medium_threshold', 30):.1f}%")
        with col2:
            st.info(f"**SHAP Samples:** {st.session_state.get('max_shap_samples', 1000):,}")
            st.info(f"**Model Status:** ✅ Ready")
    
    with tab3:
        st.markdown("""
        ### 📊 Dashboard Navigation
        
        **🏠 Home**
        - System overview and quick start guide
        - Configuration instructions
        - Current system status
        
        **📊 Turnover Prediction**
        - Upload employee dataset
        - View individual risk predictions
        - Analyze key turnover factors (SHAP)
        - Download detailed reports
        
        **📈 Organizational Insights**
        - Department-level comparisons
        - What-if scenario simulations
        - Strategic intervention planning
        """)

    # Quick Stats if predictions exist
    if 'predictions' in st.session_state:
        st.markdown("---")
        st.markdown("## 📈 Latest Analysis Summary")
        
        df = st.session_state['predictions']
        high_thresh = st.session_state.get('high_threshold', 50.0)
        medium_thresh = st.session_state.get('medium_threshold', 30.0)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "👥 Total Employees", 
                f"{len(df):,}",
                help="Total employees in the uploaded dataset"
            )
        
        with col2:
            high_risk_count = len(df[df['turnover_probability (%)'] >= high_thresh])
            high_risk_pct = (high_risk_count / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "🔴 High Risk", 
                f"{high_risk_count:,}",
                delta=f"{high_risk_pct:.1f}%",
                delta_color="inverse",
                help="Employees with high turnover risk"
            )
        
        with col3:
            avg_risk = df['turnover_probability (%)'].mean()
            st.metric(
                "📊 Average Risk", 
                f"{avg_risk:.1f}%",
                help="Mean turnover probability across all employees"
            )
        
        with col4:
            medium_risk_count = len(df[(df['turnover_probability (%)'] >= medium_thresh) & 
                                      (df['turnover_probability (%)'] < high_thresh)])
            st.metric(
                "🟡 Medium Risk", 
                f"{medium_risk_count:,}",
                help="Employees with moderate turnover risk"
            )

    st.stop()

# ============================================
# TURNOVER PREDICTION PAGE
# ============================================
elif page == "📊 Turnover Prediction":
    
    st.title("📊 Employee Turnover Prediction")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666;'>
    Predict individual employee turnover risk using AI-powered analytics. 
    Upload your employee data to get instant risk assessments and actionable insights.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Upload Section with better styling
    st.markdown("### 📤 Upload Employee Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("📋 **Upload a CSV file** containing employee information. The system will automatically process and predict turnover risk for each employee.")
    
    with col2:
        with st.expander("❓ **Need Help?**"):
            st.markdown("""
            **Tips:**
            - CSV format only
            - No 'left' column needed
            - See column guide below
            - Max file size: 200MB
            """)

    # GUIDE KOLOM dengan tampilan yang lebih menarik
    with st.expander("📋 **View Required Columns & Format Examples**", expanded=False):
        st.markdown("#### Required Columns for Prediction")
        
        # Tabel dengan styling yang lebih baik
        col_guide_df = pd.DataFrame({
            '📌 Column': ['satisfaction_level', 'last_evaluation', 'number_project', 
                         'average_montly_hours', 'time_spend_company', 'work_accident', 
                         'promotion_last_5years', 'salary'],
            '🔄 Accepted Aliases': [
                'satisfaction, kepuasan',
                'evaluation, eval_score',
                'projects, jumlah_proyek',
                'monthly_hours, jam_kerja',
                'tenure, masa_kerja',
                'accident, kecelakaan',
                'promotion, promosi',
                'gaji, compensation'
            ],
            '📊 Data Type': ['Float (0-1)', 'Float (0-1)', 'Integer', 'Integer', 
                           'Integer', 'Binary (0/1)', 'Binary (0/1)', 'Categorical'],
            '💡 Example': ['0.78', '0.85', '5', '180', '3', '0', '0', 'low/medium/high']
        })
        
        st.dataframe(col_guide_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Optional Columns (for identification)")
        st.markdown("""
        - `employee_id` / `id` / `emp_id`
        - `name` / `nama` / `employee_name`
        - `dept` / `department` / `departemen`
        """)
        
        st.success("💡 **Smart Recognition:** System automatically recognizes column variations (e.g., 'gaji' → 'salary')")

    # File uploader dengan styling
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=['csv'],
        help="Upload employee data in CSV format",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        try:
            # Load data dengan progress
            with st.spinner("📂 Loading your data..."):
                df_new = pd.read_csv(uploaded_file)
            
            # Data Preview Section
            st.markdown("### 👁️ Data Preview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{len(df_new):,}")
            with col2:
                st.metric("📋 Total Columns", f"{len(df_new.columns):,}")
            with col3:
                duplicates = df_new.duplicated().sum()
                st.metric("🔄 Duplicates", f"{duplicates:,}")
            
            st.dataframe(
                df_new.head(10), 
                use_container_width=True,
                height=300
            )
            
            # Simpan kolom untuk output
            output_cols = {}
            if 'employee_id' in df_new.columns:
                output_cols['employee_id'] = df_new['employee_id']
            if 'name' in df_new.columns:
                output_cols['name'] = df_new['name']
            if 'dept' in df_new.columns:
                output_cols['dept'] = df_new['dept']
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Prediction Button dengan styling yang lebih menarik
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                predict_button = st.button(
                    "🔮 Run Prediction Analysis", 
                    type="primary", 
                    use_container_width=True
                )
            
            if predict_button:
                with st.spinner("⚙️ Processing and predicting... Please wait"):
                    progress_bar = st.progress(0)
                    
                    # Preprocessing
                    progress_bar.progress(25)
                    df_processed = preprocess_data(df_new, is_training=False)
                    
                    # Align features
                    progress_bar.progress(50)
                    df_features = align_features(df_processed, feature_list)
                    
                    # Predict
                    progress_bar.progress(75)
                    predictions = model.predict_proba(df_features)[:, 1] * 100
                    
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
                    
                    # Sort by risk
                    result_df = result_df.sort_values('turnover_probability (%)', ascending=False).reset_index(drop=True)
                    
                    # Calculate SHAP
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
                    
                    progress_bar.progress(100)
                
                st.success("✅ Prediction completed successfully!")
                st.balloons()
            
        except Exception as e:
            st.error(f"❌ **Error processing file:** {str(e)}")
            with st.expander("🔍 View Error Details"):
                st.exception(e)
            st.stop()

    # ============================================
    # DISPLAY PREDICTIONS
    # ============================================
    if 'predictions' in st.session_state:
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🎯 Prediction Results")
        
        predictions_df = st.session_state['predictions']
        
        # Metrics dengan styling yang lebih menarik
        high_thresh = st.session_state.get('high_threshold', 50.0)
        medium_thresh = st.session_state.get('medium_threshold', 30.0)

        # Risk Distribution
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_emp = len(predictions_df)
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{:,}</h3>
                <p style='margin: 0.5rem 0 0 0;'>👥 Total Employees</p>
            </div>
            """.format(total_emp), unsafe_allow_html=True)
        
        with col2:
            high_risk = len(predictions_df[predictions_df['turnover_probability (%)'] >= high_thresh])
            high_pct = (high_risk / total_emp * 100) if total_emp > 0 else 0
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{:,}</h3>
                <p style='margin: 0.5rem 0 0 0;'>🔴 High Risk ({:.1f}%)</p>
            </div>
            """.format(high_risk, high_pct), unsafe_allow_html=True)
        
        with col3:
            medium_risk = len(predictions_df[(predictions_df['turnover_probability (%)'] >= medium_thresh) & 
                                            (predictions_df['turnover_probability (%)'] < high_thresh)])
            medium_pct = (medium_risk / total_emp * 100) if total_emp > 0 else 0
            st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{:,}</h3>
                <p style='margin: 0.5rem 0 0 0;'>🟡 Medium Risk ({:.1f}%)</p>
            </div>
            """.format(medium_risk, medium_pct), unsafe_allow_html=True)
        
        with col4:
            low_risk = len(predictions_df[predictions_df['turnover_probability (%)'] < medium_thresh])
            low_pct = (low_risk / total_emp * 100) if total_emp > 0 else 0
            st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                <h3 style='color: white; margin: 0; font-size: 2rem;'>{:,}</h3>
                <p style='margin: 0.5rem 0 0 0;'>🟢 Low Risk ({:.1f}%)</p>
            </div>
            """.format(low_risk, low_pct), unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Employee Ranking Table
        st.markdown("### 📋 Employee Risk Ranking")
        st.caption("Employees are ranked by turnover probability (highest risk first)")
        
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
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🔬 Key Turnover Factors (SHAP Analysis)")
        
        # Info sampling dengan styling lebih baik
        sample_info = st.session_state.get('shap_sample_info', {})
        if sample_info.get('sampled', False):
            st.info(
                f"📊 **Analysis Method:** Stratified sampling | "
                f"**Samples Used:** {sample_info['sample_size']:,} of {sample_info['total_data']:,} employees | "
                f"**Confidence:** High (representative sample)"
            )
        else:
            st.info(f"📊 **Analysis Method:** Full dataset analysis | **Total Employees:** {sample_info.get('total_data', 'N/A'):,}")
        
        feature_imp = st.session_state['feature_importance']
        shap_values = st.session_state['shap_values']
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("#### 📊 Top 10 Influential Factors")
            
            # Create a bar chart for better visualization
            top10 = feature_imp.head(10)
            fig = px.bar(
                top10, 
                x='Pengaruh Terhadap Risiko Turnover', 
                y='feature',
                orientation='h',
                color='Pengaruh Terhadap Risiko Turnover',
                color_continuous_scale='Blues',
                labels={'Pengaruh Terhadap Risiko Turnover': 'Impact Score', 'feature': 'Factor'}
            )
            fig.update_layout(
                showlegend=False,
                height=400,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🏆 Top 3 Critical Factors")
            top3 = feature_imp.head(3)
            
            for idx, (i, row) in enumerate(top3.iterrows(), 1):
                if idx == 1:
                    badge = "🥇"
                    color = "#FFD700"
                elif idx == 2:
                    badge = "🥈"
                    color = "#C0C0C0"
                else:
                    badge = "🥉"
                    color = "#CD7F32"
                
                st.markdown(f"""
                <div style='background-color: {color}20; border-left: 4px solid {color}; 
                            padding: 1rem; margin: 0.5rem 0; border-radius: 5px;'>
                    <h4 style='margin: 0; color: #333;'>{badge} {row['feature'].replace('_', ' ').title()}</h4>
                    <p style='margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: bold; color: {color};'>
                        {row['Pengaruh Terhadap Risiko Turnover']:.4f}
                    </p>
                    <p style='margin: 0; font-size: 0.85rem; color: #666;'>Impact Score</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ============================================
        # RECOMMENDATIONS
        # ============================================
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 💡 Strategic Recommendations")
        st.caption("Action plans based on your top turnover factors")

        top3_features = feature_imp.head(3)['feature'].tolist()

        # Mapping rekomendasi
        recommendation_map = {
            "satisfaction_level": {
                "icon": "😊",
                "title": "Employee Satisfaction Enhancement",
                "action": "Implement regular feedback sessions, improve workload distribution, and strengthen manager-employee relationships.",
                "impact": "Significant improvement in retention and employee commitment.",
                "priority": "High"
            },
            "average_montly_hours": {
                "icon": "⏰",
                "title": "Workload & Work-Life Balance",
                "action": "Establish healthy work hours policy and ensure adequate rest periods.",
                "impact": "Reduces burnout, fatigue, and resignation intentions.",
                "priority": "High"
            },
            "time_spend_company": {
                "icon": "🎯",
                "title": "Career Development & Recognition",
                "action": "Provide rotation opportunities, skill training, and formal performance recognition.",
                "impact": "Increases loyalty among both new and tenured employees.",
                "priority": "Medium"
            },
            "number_project": {
                "icon": "📊",
                "title": "Task Allocation Optimization",
                "action": "Ensure project assignments match capacity with weekly workload reviews.",
                "impact": "Reduces excessive stress and increases engagement.",
                "priority": "Medium"
            },
            "promotion_last_5years": {
                "icon": "🚀",
                "title": "Career Equity & Advancement",
                "action": "Review promotion fairness and improve performance evaluation transparency.",
                "impact": "Enhances sense of fairness and work motivation.",
                "priority": "High"
            },
            "salary_encoded": {
                "icon": "💰",
                "title": "Competitive Compensation & Benefits",
                "action": "Conduct salary benchmarking and implement performance-based rewards.",
                "impact": "Reduces turnover due to compensation dissatisfaction.",
                "priority": "High"
            },
            "work_accident": {
                "icon": "🛡️",
                "title": "Workplace Safety & Health",
                "action": "Strengthen safety standards and provide comprehensive safety training.",
                "impact": "Improves security and trust in the organization.",
                "priority": "High"
            }
        }

        # Display recommendations in cards
        for i, feature in enumerate(top3_features, 1):
            rec = recommendation_map.get(feature, None)
            
            if rec:
                priority_color = "#dc3545" if rec['priority'] == "High" else "#ffc107"
                
                with st.expander(f"**{i}. {rec['icon']} {rec['title']}**", expanded=(i==1)):
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"""
                        **🎯 Root Cause Factor:** `{feature}`
                        
                        **📋 Recommended Actions:**
                        {rec['action']}
                        
                        **💼 Business Impact:**
                        {rec['impact']}
                        """)
                    
                    with col_b:
                        st.markdown(f"""
                        <div style='background-color: {priority_color}20; border: 2px solid {priority_color}; 
                                    padding: 1rem; border-radius: 8px; text-align: center;'>
                            <p style='margin: 0; font-weight: bold; color: {priority_color};'>Priority</p>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;'>{rec['priority']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                with st.expander(f"**{i}. {feature.replace('_',' ').title()}**"):
                    st.info(f"**Factor:** `{feature}`\n\nCustom recommendations can be developed through further HR analysis.")

        
        # ============================================
        # DOWNLOAD RESULTS
        # ============================================
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 📥 Export Results")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Prediction Results (CSV)",
                data=csv,
                file_name="employee_turnover_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )

# ==========================
# ORGANIZATIONAL INSIGHTS PAGE
# ==========================
elif page == "📈 Organizational Insights":

    st.title("📈 Organizational Insights & Strategy Simulation")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666;'>
    Analyze turnover risk at organizational level and simulate the impact of strategic interventions.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Pastikan prediksi sudah ada
    if 'predictions' not in st.session_state:
        st.warning("⚠️ **No prediction data available.** Please run predictions first in the 'Turnover Prediction' tab.")
        st.info("👉 Go to **📊 Turnover Prediction** → Upload your employee data → Run prediction")
        st.stop()

    predictions_df = st.session_state['predictions']
    feature_imp = st.session_state['feature_importance']

    # ======== Department Comparison ========
    st.markdown("## 🏢 Department Risk Analysis")
    st.caption("Compare turnover risk across different departments")

    if 'dept' in predictions_df.columns:
        dept_risk = predictions_df.groupby('dept')['turnover_probability (%)'].agg(['mean', 'count']).reset_index()
        dept_risk.columns = ['Department', 'Average Risk (%)', 'Employee Count']
        dept_risk = dept_risk.sort_values('Average Risk (%)', ascending=False)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Visualization
            fig = px.bar(
                dept_risk,
                x='Department',
                y='Average Risk (%)',
                color='Average Risk (%)',
                color_continuous_scale='RdYlGn_r',
                text='Average Risk (%)',
                title='Average Turnover Risk by Department'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Department Stats")
            st.dataframe(
                dept_risk.style.background_gradient(
                    subset=['Average Risk (%)'],
                    cmap='RdYlGn_r'
                ).format({'Average Risk (%)': '{:.2f}%'}),
                use_container_width=True,
                hide_index=True,
                height=400
            )

        # Highlight highest risk department
        highest = dept_risk.iloc[0]
        st.warning(
            f"🎯 **Priority Focus:** The **{highest['Department']}** department shows the highest average "
            f"turnover risk at **{highest['Average Risk (%)']:.2f}%** "
            f"({int(highest['Employee Count'])} employees). Immediate intervention recommended."
        )

    else:
        st.error("❌ Department information not available in the dataset. Please include 'dept' column in your data.")
    
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ======== Simulasi Intervensi Strategi ========
    st.markdown("## 🔬 What-If Scenario Simulation")
    st.markdown("""
    <p style='font-size: 1.05rem; color: #666;'>
    Test different intervention strategies and see their potential impact on organizational turnover risk.
    Adjust key factors and simulate how changes would affect your workforce.
    </p>
    """, unsafe_allow_html=True)

    # Pastikan prerequisites tersedia
    if 'model' not in st.session_state or 'feature_list' not in st.session_state or 'df_features' not in st.session_state:
        st.warning("⚠️ Required data not available. Please run predictions first.")
        st.stop()

    # Load variables from session_state
    model = st.session_state['model']
    feature_list = st.session_state['feature_list']
    df_feat_source = st.session_state['df_features'].copy()

    if df_feat_source is not None:
        # Top features dari SHAP
        feature_imp = st.session_state.get('feature_importance', None)
        if feature_imp is not None:
            top_features = feature_imp.head(5)['feature'].tolist()
        else:
            top_features = [c for c in df_feat_source.columns if c in feature_list][:5]

        st.markdown("### 🎛️ Configure Your Intervention Strategy")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Select Factors to Modify")
            sel_top = st.multiselect(
                "Choose from top influential factors:",
                top_features,
                default=top_features[:1],
                help="Select one or more factors to simulate changes"
            )
        
        with col2:
            st.markdown("#### Or Add Custom Factor")
            manual_feature = st.text_input(
                "Enter factor name:",
                "",
                help="Leave empty to skip",
                placeholder="e.g., satisfaction_level"
            )

        # Build interventions
        interventions = []
        chosen = sel_top.copy()
        if manual_feature.strip():
            chosen.append(manual_feature.strip())

        if chosen:
            st.markdown("### ⚙️ Define Changes for Selected Factors")
            st.caption("Specify how you want to modify each factor")
            
            for f in chosen:
                with st.expander(f"**{f.replace('_', ' ').title()}**", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        typ = st.selectbox(
                            f"Change type for {f}",
                            options=['add', 'set', 'scale'],
                            key=f+"_typ",
                            help="Add: increase/decrease by value | Set: change to exact value | Scale: multiply by factor"
                        )
                    
                    with col_b:
                        if typ == 'add':
                            val = st.number_input(
                                f"Add value (e.g., +0.10):",
                                value=0.10,
                                step=0.01,
                                key=f+"_val",
                                format="%.2f"
                            )
                        elif typ == 'set':
                            val = st.number_input(
                                f"Set to value:",
                                value=1.0,
                                step=0.01,
                                key=f+"_val",
                                format="%.2f"
                            )
                        else:
                            val = st.number_input(
                                f"Multiply by (e.g., 0.9):",
                                value=0.9,
                                step=0.01,
                                key=f+"_val",
                                format="%.2f"
                            )
                    
                    interventions.append({'feature': f, 'type': typ, 'value': float(val)})
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Run Simulation Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                run_sim = st.button(
                    "🚀 Run Simulation",
                    type="primary",
                    use_container_width=True
                )
            
            if run_sim:
                high_thresh = st.session_state.get('high_threshold', 50.0)
                medium_thresh = st.session_state.get('medium_threshold', 30.0)

                with st.spinner("⚙️ Calculating simulation results..."):
                    try:
                        result = run_intervention_simulation(
                            df_feat_source, model, interventions,
                            feature_list, high_thresh, medium_thresh
                        )
                    except Exception as e:
                        st.error(f"❌ Simulation failed: {e}")
                        result = None

                if result:
                    st.success("✅ Simulation completed successfully!")
                    
                    st.markdown("---")
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("## 📊 Simulation Results")
                    
                    # Summary Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1.5rem; border-radius: 10px; color: white;'>
                            <h4 style='color: white; margin: 0;'>📉 Average Risk Change</h4>
                            <h2 style='color: white; margin: 0.5rem 0;'>{:.2f}%</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Before: {:.2f}% → After: {:.2f}%</p>
                        </div>
                        """.format(result['avg_drop'], result['old_avg'], result['new_avg']),
                        unsafe_allow_html=True)
                    
                    with col2:
                        high_change = result['new_counts']['high'] - result['old_counts']['high']
                        change_symbol = "↓" if high_change < 0 else "↑" if high_change > 0 else "→"
                        change_color = "#00ff00" if high_change < 0 else "#ff0000" if high_change > 0 else "#ffff00"
                        
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                    padding: 1.5rem; border-radius: 10px; color: white;'>
                            <h4 style='color: white; margin: 0;'>🔴 High Risk Employees</h4>
                            <h2 style='color: white; margin: 0.5rem 0;'>{} → {}</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Change: <span style='color: {};'>{} {}</span></p>
                        </div>
                        """.format(
                            result['old_counts']['high'],
                            result['new_counts']['high'],
                            change_color,
                            change_symbol,
                            abs(high_change)
                        ), unsafe_allow_html=True)
                    
                    with col3:
                        total_reduction = (result['old_counts']['high'] + result['old_counts']['medium']) - \
                                        (result['new_counts']['high'] + result['new_counts']['medium'])
                        
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                                    padding: 1.5rem; border-radius: 10px; color: white;'>
                            <h4 style='color: white; margin: 0;'>🎯 Total At-Risk Reduction</h4>
                            <h2 style='color: white; margin: 0.5rem 0;'>{}</h2>
                            <p style='margin: 0; font-size: 0.9rem;'>Employees moved to low risk</p>
                        </div>
                        """.format(total_reduction), unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed Breakdown
                    st.markdown("### 📋 Detailed Category Changes")
                    
                    breakdown_df = pd.DataFrame({
                        'Risk Category': ['🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk'],
                        'Before': [
                            result['old_counts']['high'],
                            result['old_counts']['medium'],
                            result['old_counts']['low']
                        ],
                        'After': [
                            result['new_counts']['high'],
                            result['new_counts']['medium'],
                            result['new_counts']['low']
                        ],
                        'Change': [
                            result['new_counts']['high'] - result['old_counts']['high'],
                            result['new_counts']['medium'] - result['old_counts']['medium'],
                            result['new_counts']['low'] - result['old_counts']['low']
                        ]
                    })
                    
                    st.dataframe(
                        breakdown_df.style.apply(
                            lambda x: ['background-color: #ffebee' if v > 0 else 'background-color: #e8f5e9' if v < 0 else '' 
                                      for v in x], subset=['Change']
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Most Impacted Employees
                    st.markdown("### 👥 Most Impacted Employees")
                    st.caption("Top 10 employees with largest probability changes")
                    
                    deltas = pd.Series(result['delta_perc'])
                    top_impact_idx = deltas.abs().nlargest(10).index
                    
                    impact_table = pd.DataFrame({
                        'Employee Index': top_impact_idx,
                        'Before (%)': result['old_proba'][top_impact_idx].round(2),
                        'After (%)': result['new_proba'][top_impact_idx].round(2),
                        'Change (%)': result['delta_perc'][top_impact_idx].round(2)
                    })
                    
                    st.dataframe(
                        impact_table.style.background_gradient(
                            subset=['Change (%)'],
                            cmap='RdYlGn',
                            vmin=-20,
                            vmax=20
                        ),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Interpretation
                    st.markdown("### 💡 Interpretation")
                    
                    if result['avg_drop'] > 0:
                        st.success(
                            f"✅ **Positive Impact:** This intervention strategy would reduce average turnover "
                            f"risk by **{result['avg_drop']:.2f} percentage points**. "
                            f"This is a promising strategy worth implementing."
                        )
                    elif result['avg_drop'] < 0:
                        st.error(
                            f"⚠️ **Negative Impact:** This intervention would increase average turnover risk "
                            f"by **{abs(result['avg_drop']):.2f} percentage points**. "
                            f"Consider revising the strategy."
                        )
                    else:
                        st.info("ℹ️ **Neutral Impact:** This intervention shows minimal effect on turnover risk.")
        
        else:
            st.info("👆 Please select at least one factor to simulate changes.")

# ============================================
# FOOTER
# ============================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0; color: #666;'>
    <p style='margin: 0; font-size: 0.9rem;'>
        🏢 <strong>Employee Turnover Prediction System</strong> | 
        Powered by Machine Learning & SHAP Analysis
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
        Developed by IT/Data Team | Version 2.0 | © 2024
    </p>
</div>
""", unsafe_allow_html=True)