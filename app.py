# streamlit_turnover_app.py
# Streamlit app: Full pipeline from preprocessing (user-provided) -> Random Forest -> DSS outputs

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pickle

st.set_page_config(page_title="Turnover Prediction DSS", layout="wide")

st.title("Turnover Prediction — Decision Support System (DSS)")
st.write("Upload file HR.csv (hasil preprocessing akan dijalankan otomatis). App ini melakukan: preprocessing (duplikasi/winsorize/encoding), split data, train Random Forest, evaluasi, feature importance, probabilitas turnover, dan rekomendasi manajerial.")

# ------------------------- Helper: preprocessing (user's code integrated) -------------------------

def run_preprocessing(df_input):
    df = df_input.copy()

    # 1. CEK & HAPUS DUPLIKASI
    st.text("=== 1. CLEANING DUPLIKASI ===")
    dup_before = int(df.duplicated().sum())
    st.text(f"Jumlah duplikasi sebelum dibersihkan: {dup_before}")

    df = df.drop_duplicates()

    dup_after = int(df.duplicated().sum())
    st.text(f"Jumlah duplikasi setelah dibersihkan: {dup_after}\n")

    # 2. WINSORIZING untuk time_spend_company
    if 'time_spend_company' in df.columns:
        st.text("=== 2. WINSORIZING time_spend_company ===")
        Q1 = df['time_spend_company'].quantile(0.25)
        Q3 = df['time_spend_company'].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df['time_spend_company_winsor'] = np.where(
            df['time_spend_company'] > upper, upper,
            np.where(df['time_spend_company'] < lower, lower, df['time_spend_company'])
        )

        st.text("Contoh 5 baris hasil winsorizing:")
        st.write(df[['time_spend_company', 'time_spend_company_winsor']].head(5))
        st.text("")
    else:
        st.warning("Kolom 'time_spend_company' tidak ditemukan — melewati langkah winsorizing.")

    # 3. LABEL ENCODING untuk dept & salary
    st.text("=== 3. ENCODING dept & salary ===")

    # salary mapping (ordinal)
    if 'salary' in df.columns:
        salary_mapping = {"low": 1, "medium": 2, "high": 3}
        df['salary_encoded'] = df['salary'].map(salary_mapping)
        st.text(f"Mapping Salary: {salary_mapping}")
    else:
        st.warning("Kolom 'salary' tidak ditemukan — pastikan dataset memiliki kolom salary")

    # one-hot department
    if 'dept' in df.columns:
        dept_dummies = pd.get_dummies(df['dept'], prefix='dept', dtype=int)
        df = pd.concat([df, dept_dummies], axis=1)
        st.text(f"Dept One-Hot Columns: {list(dept_dummies.columns)}")
    else:
        st.warning("Kolom 'dept' tidak ditemukan — melewati one-hot encoding untuk dept.")

    # 4. DROP kolom asli dept & salary
    drop_cols = [c for c in ['dept', 'salary'] if c in df.columns]
    if drop_cols:
        df_pre = df.drop(drop_cols, axis=1)
    else:
        df_pre = df.copy()

    st.text("\n===== PREPROCESSING SELESAI =====")
    st.text(f"Shape akhir dataset: {df_pre.shape}")
    st.write(df_pre.head(5))

    return df_pre


# ------------------------- Sidebar: upload or use example -------------------------
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload HR.csv (hasil preprocessing atau mentah) ", type=['csv'])
use_sample = st.sidebar.checkbox("Gunakan contoh kecil (demo)")

if uploaded_file is None and not use_sample:
    st.info("Silakan upload file HR.csv yang sudah/prereprocessed atau centang 'Gunakan contoh kecil'.")

# demo sample if user wants
if use_sample:
    # create a tiny synthetic demo dataframe
    demo = pd.DataFrame({
        'employee_id': range(1,21),
        'name': [f'Karyawan_{i}' for i in range(1,21)],
        'satisfaction_level': np.random.rand(20),
        'last_evaluation': np.random.rand(20),
        'number_project': np.random.randint(2,7,20),
        'average_monthly_hours': np.random.randint(90,310,20),
        'time_spend_company': np.random.randint(1,11,20),
        'work_accident': np.random.choice([0,1],20, p=[0.95,0.05]),
        'promotion_last_5years': np.random.choice([0,1],20, p=[0.9,0.1]),
        'dept': np.random.choice(['sales','technical','hr','support'],20),
        'salary': np.random.choice(['low','medium','high'],20, p=[0.6,0.3,0.1]),
        'left': np.random.choice([0,1],20, p=[0.8,0.2])
    })
    df_raw = demo
    st.sidebar.success("Menggunakan dataset demo kecil — tidak perlu upload file.")

elif uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.sidebar.success("File berhasil diupload.")
    except Exception as e:
        st.error("Gagal membaca file CSV: " + str(e))
        st.stop()

else:
    df_raw = None

# ------------------------- Run preprocessing -------------------------
if df_raw is not None:
    with st.spinner('Menjalankan preprocessing...'):
        df_preprocessed = run_preprocessing(df_raw)

    # save cleaned CSV to allow download
    buf = io.StringIO()
    df_preprocessed.to_csv(buf, index=False)
    csv_bytes = buf.getvalue().encode()
    st.download_button("Download cleaned_dataset.csv", data=csv_bytes, file_name='cleaned_dataset.csv')

    # ------------------------- Prepare X & y -------------------------
    if 'left' not in df_preprocessed.columns:
        st.error("Kolom 'turnover' tidak ditemukan. Pastikan ada kolom target bernama 'turnover' (0/1).")
        st.stop()

    X = df_preprocessed.drop('left', axis=1)
    y = df_preprocessed['left']

    # If name/employee_id exist, keep them for final display — but remove from X used for training
    id_cols = []
    for c in ['name', 'employee_id', 'id']:
        if c in X.columns:
            id_cols.append(c)

    X_model = X.drop(columns=id_cols, errors='ignore')

    # Convert any non-numeric columns that remain (should be none after preprocessing)
    non_numeric = X_model.select_dtypes(include=['object', 'category']).columns.tolist()
    if non_numeric:
        st.warning(f"Masih terdapat kolom non-numeric yang akan di-encode otomatis: {non_numeric}")
        X_model = pd.get_dummies(X_model, drop_first=True)

    st.subheader("Modeling options")
    test_size = st.slider("Test set size (fraction)", min_value=0.1, max_value=0.4, value=0.2)
    n_estimators = st.number_input("Random Forest - n_estimators", min_value=50, max_value=1000, value=200, step=50)
    random_state = 42

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, random_state=random_state, stratify=y)

    # Fit model
    rf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=random_state)
    rf.fit(X_train, y_train)

    # Predict & probabilities
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📊 Model performance")
    col1, col2 = st.columns([1,1])
    with col1:
        st.metric("Accuracy", f"{acc:.3f}")
        st.write(pd.DataFrame(clf_report).transpose())

    with col2:
        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

    # Feature importance
    st.subheader("⭐ Feature importance")
    feat_imp = pd.DataFrame({'feature': X_model.columns, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)
    st.write(feat_imp.head(20))

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.barh(feat_imp['feature'].iloc[::-1], feat_imp['importance'].iloc[::-1])
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature importance (Random Forest)')
    st.pyplot(fig2)

    # Determine top feature (human readable)
    top_feature = feat_imp.iloc[0]['feature']

    # DSS: create results table with id columns if exist
    results = X_test.copy()
    results['turnover_prob'] = y_prob
    results['prediction'] = y_pred

    # re-attach id columns from original X if they existed
    if id_cols:
        # We need to align indices — X_test has indices from original dataset
        for c in id_cols:
            if c in X.columns:
                results[c] = X.loc[results.index, c]

    # Sorting by probability
    results_sorted = results.sort_values(by='turnover_prob', ascending=False)

    st.subheader("🔎 Top employees by turnover probability")
    display_cols = id_cols + ['turnover_prob', 'prediction'] + [c for c in results_sorted.columns if c not in id_cols + ['turnover_prob','prediction']]
    # show top 10
    st.write(results_sorted[display_cols].head(10))

    # Recommendation logic based on top feature
    st.subheader("💡 Insight & Rekomendasi Manajerial (otomatis)")
    tf = top_feature.lower()
    if 'salary' in tf or tf == 'salary_encoded':
        st.markdown("*Insight:* Salary / kompensasi adalah faktor paling berpengaruh terhadap turnover pada model ini.")
        st.markdown("*Rekomendasi:* Lakukan audit salary band, pertimbangkan penyesuaian gaji untuk kelompok berisiko tinggi, program reward/bonus, dan benchmarking pasar.")
    elif tf.startswith('dept_') or 'dept' in tf:
        st.markdown("*Insight:* Department tertentu berkontribusi besar terhadap turnover.")
        st.markdown("*Rekomendasi:* Tinjau beban kerja, kepemimpinan, dan kultur di department bersangkutan; program retensi khusus departemen.")
    elif 'satisfaction' in tf or 'satis' in tf:
        st.markdown("*Insight:* Tingkat kepuasan (satisfaction) sangat mempengaruhi turnover.")
        st.markdown("*Rekomendasi:* Survei kepuasan lebih dalam, focus group, perbaiki masalah yang sering muncul (workload, recognition, career path).")
    else:
        st.markdown(f"*Insight:* Fitur paling berpengaruh: *{top_feature}*")
        st.markdown("*Rekomendasi umum:* Analisa lebih lanjut faktor ini, pertimbangkan intervensi kebijakan HR yang spesifik (kompensasi, workload, training, dsb).")

    # Allow download of predictions (top -> full)
    out_buf = io.StringIO()
    export_df = results_sorted.copy()
    export_df.to_csv(out_buf, index=False)
    st.download_button("Download predictions (CSV)", data=out_buf.getvalue().encode(), file_name='turnover_predictions.csv')

    # Save trained model as pickle
    model_bytes = pickle.dumps(rf)
    st.download_button("Download trained RandomForest model (.pkl)", data=model_bytes, file_name='rf_turnover_model.pkl')

    st.success("Selesai — gunakan tabel top karyawan dan rekomendasi untuk prioritas tindakan HR.")

# End

else:
    st.stop()