import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np
import joblib 

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Encoding
le = LabelEncoder()
df['OverTime'] = le.fit_transform(df['OverTime'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['Department'] = le.fit_transform(df['Department'])
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

# Sidebar
st.sidebar.title("📊 Dashboard HR Analytics")
menu = st.sidebar.radio("Pilih Menu:", ["📌 Prediksi Attrition", "💰 Prediksi Gaji", "👥 Segmentasi Karyawan"])

# === 1. Attrition (Klasifikasi) ===
if menu == "📌 Prediksi Attrition":
    st.title("📌 Prediksi Karyawan Akan Resign atau Tidak")

    # Input data
    age = st.slider("Umur", 18, 60, 30)
    distance = st.slider("Jarak dari rumah ke kantor", 1, 30, 5)
    job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    monthly_income = st.number_input("Pendapatan Bulanan", min_value=0, max_value=100_000_000_000, step=1)
    years = st.slider("Lama bekerja di perusahaan", 0, 40, 3)
    overtime = st.selectbox("Lembur?", ["Tidak", "Ya"])
    jobrole = st.selectbox("Jabatan", sorted(df['JobRole'].unique()))
    dept = st.selectbox("Departemen", sorted(df['Department'].unique()))

    input_data = pd.DataFrame([[age, distance, job_sat, env_sat, monthly_income,
                                years, 1 if overtime == "Ya" else 0,
                                jobrole, dept]],
                              columns=['Age', 'DistanceFromHome', 'JobSatisfaction',
                                       'EnvironmentSatisfaction', 'MonthlyIncome',
                                       'YearsAtCompany', 'OverTime', 'JobRole', 'Department'])

    # Model
    model = RandomForestClassifier()
    X = df[['Age', 'DistanceFromHome', 'JobSatisfaction', 'EnvironmentSatisfaction',
            'MonthlyIncome', 'YearsAtCompany', 'OverTime', 'JobRole', 'Department']]
    y = df['Attrition']
    model.fit(X, y)

    pred = model.predict(input_data)[0]
    # Feature importance

    # Output prediksi
    st.subheader("📢 Hasil Prediksi:")
    if pred == 1:
        st.error("⚠️ Karyawan ini berpotensi **AKAN** resign.")
    else:
        st.success("✅ Karyawan ini kemungkinan **TIDAK AKAN** resign.")

    # Alasan (feature importance)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_feats = [(input_data.columns[i], input_data.iloc[0, i], importances[i]) for i in sorted_idx[:3]]

    st.markdown("### 🔍 Alasan Prediksi")

    def interpret_reason(feat, val):
        if feat == "MonthlyIncome":
            return "karena gajinya tergolong rendah" if val < 3000 else "karena gajinya cukup tinggi"
        elif feat == "OverTime":
            return "karena sering lembur" if val == 1 else "karena tidak lembur"
        elif feat == "Age":
            return "karena usia sudah mendekati pensiun" if val >= 55 else "karena usia masih produktif"
        elif feat == "DistanceFromHome":
            return "karena jarak rumah ke kantor cukup jauh" if val > 15 else "karena jarak rumah dekat"
        elif feat == "JobSatisfaction":
            return "karena tingkat kepuasan kerja rendah" if val <= 2 else "karena cukup puas dengan pekerjaannya"
        elif feat == "EnvironmentSatisfaction":
            return "karena tidak nyaman dengan lingkungan kerja" if val <= 2 else "karena cukup nyaman dengan lingkungan kerja"
        elif feat == "YearsAtCompany":
            return "karena belum lama bekerja" if val < 2 else "karena sudah cukup lama bekerja"
        else:
            return "berkontribusi pada keputusan"

    for feat, val, score in top_feats:
        reason = interpret_reason(feat, val)
        st.markdown(f"- **{feat} = {val}** → {reason} *(importance: {score:.2f})*")

    # Visualisasi pie chart attrition
    st.subheader("📊 Distribusi Karyawan Resign vs Tidak")
    attrition_counts = df['Attrition'].value_counts().rename({0: "Tidak Resign", 1: "Resign"})
    fig = px.pie(names=attrition_counts.index, values=attrition_counts.values,
                 title="Distribusi Karyawan")
    st.plotly_chart(fig)


# === 2. Prediksi Gaji (Regresi) ===
elif menu == "💰 Prediksi Gaji":
    st.title("💰 Estimasi Gaji Karyawan")

    # Encoder konsisten
    le_job = LabelEncoder()
    df['JobRole'] = le_job.fit_transform(df['JobRole'])

    # Input dari user
    jobrole_str = st.selectbox("Jabatan", le_job.classes_)
    jobrole = le_job.transform([jobrole_str])[0]
    joblevel = st.selectbox("Level Jabatan", [1, 2, 3, 4, 5])
    education = st.selectbox("Tingkat Pendidikan", [1, 2, 3, 4, 5])
    years_company = st.slider("Lama di Perusahaan (tahun)", 0, 40, 5)
    rating = st.selectbox("Performance Rating", [1, 2, 3, 4])
    total_years = st.slider("Total Pengalaman Kerja", 0, 40, 6)

    # Bentuk input
    input_data = pd.DataFrame([[joblevel, education, years_company, rating, total_years, jobrole]],
        columns=['JobLevel', 'Education', 'YearsAtCompany', 'PerformanceRating', 'TotalWorkingYears', 'JobRole']
    )

    # Model regresi dari data yang sudah dinormalisasi
    from sklearn.ensemble import RandomForestRegressor

    X = df[['JobLevel', 'Education', 'YearsAtCompany', 'PerformanceRating', 'TotalWorkingYears', 'JobRole']]
    y = df['MonthlyIncome']  # ini sudah dinormalisasi (0–1)
    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X, y)

    # Prediksi
    salary_pred = reg_model.predict(input_data)[0]

    # Denormalisasi
    gaji_min = df['MonthlyIncome'].min()
    gaji_max = df['MonthlyIncome'].max()
    gaji_asli = salary_pred * (20000000 - 3000000) + 3000000
    st.success(f"💵 Estimasi Gaji Bulanan: Rp {int(gaji_asli):,}")

    # Feature importance
    importances = reg_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_feats = [(input_data.columns[i], input_data.iloc[0, i], importances[i]) for i in sorted_idx[:3]]

    st.markdown("### 🔍 Faktor Utama Penentu Gaji")

    def interpret_reason(feat, val):
        if feat == "JobLevel":
            return "karena level jabatan cukup tinggi" if val >= 3 else "karena level jabatan masih rendah"
        elif feat == "TotalWorkingYears":
            return "karena pengalaman kerja panjang" if val >= 10 else "karena pengalaman kerja masih sedikit"
        elif feat == "PerformanceRating":
            return "karena performa kerja sangat baik" if val == 4 else "karena performa kerja biasa saja"
        elif feat == "Education":
            return "karena tingkat pendidikan tinggi" if val >= 4 else "karena pendidikan standar"
        elif feat == "YearsAtCompany":
            return "karena sudah lama bekerja di perusahaan" if val >= 10 else "karena masih baru"
        else:
            return "berkontribusi terhadap estimasi gaji"

    for feat, val, score in top_feats:
        reason = interpret_reason(feat, val)
        st.markdown(f"- **{feat} = {val}** → {reason} *(importance: {score:.2f})*")

    # Visualisasi
    st.subheader("📊 Distribusi Gaji terhadap Pengalaman")
    import plotly.express as px
    fig2 = px.scatter(df, x='TotalWorkingYears', y='MonthlyIncome', color='JobLevel',
                      title='Gaji Normalisasi vs Total Pengalaman Kerja',
                      labels={'TotalWorkingYears': 'Total Working Years', 'MonthlyIncome': 'Monthly Income (0–1)'})
    st.plotly_chart(fig2)

# === 3. Clustering ===
elif menu == "👥 Segmentasi Karyawan":
    st.title("👥 Segmentasi Karyawan")

    cols = ['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)

    # Visualisasi 2D
    st.subheader("📌 Visualisasi Cluster (Income vs Masa Kerja)")
    fig3 = px.scatter(df, x='MonthlyIncome', y='YearsAtCompany',
                      color='Cluster', hover_data=['JobRole', 'Department'],
                      title='Cluster Berdasarkan Income dan Masa Kerja')
    st.plotly_chart(fig3)

    # Bar chart rata-rata gaji per cluster
    st.subheader("📈 Rata-rata Gaji per Cluster")
    avg_income = df.groupby('Cluster')['MonthlyIncome'].mean().reset_index()
    fig4 = px.bar(avg_income, x='Cluster', y='MonthlyIncome',
                  title='Rata-rata Gaji per Cluster',
                  labels={'MonthlyIncome': 'Monthly Income'})
    st.plotly_chart(fig4)

    # Boxplot distribusi umur
    st.subheader("📦 Penyebaran Usia per Cluster")
    fig5 = px.box(df, x='Cluster', y='Age', points="all", title="Distribusi Usia di Tiap Cluster")
    st.plotly_chart(fig5)

    # Statistik tabel
    st.subheader("📊 Statistik per Cluster")
    st.dataframe(df.groupby('Cluster')[cols].mean().round(2))
