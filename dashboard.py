import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px
import numpy as np

# ============================
# 📥 LOAD & PREPROCESS DATA
# ============================

df = pd.read_csv('HR_Analytics.csv')

# Encode hanya kolom biner OverTime
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# Biarkan JobRole & Department tetap string (tidak encode)
df['Attrition'] = df['Attrition'].replace({'Yes': 1, 'No': 0})  # Target binary

# ============================
# 📊 SIDEBAR MENU
# ============================

st.sidebar.title("📊 Dashboard HR Analytics")
menu = st.sidebar.radio("Pilih Menu:", ["📌 Prediksi Attrition", "💰 Prediksi Gaji", "👥 Segmentasi Karyawan"])

# ============================
# 1️⃣ PREDIKSI ATTRITION
# ============================

if menu == "📌 Prediksi Attrition":
    st.title("📌 Prediksi Karyawan Resign")

    # Form Input
    age = st.slider("Umur", 18, 60, 30)
    distance = st.slider("Jarak Rumah ke Kantor (km)", 1, 50, 10)
    income = st.number_input("Pendapatan Bulanan (Rp)", min_value=1000000, max_value=100_000_000, value=7000000, step=500000)
    overtime = st.radio("Lembur?", ["Yes", "No"])
    years = st.slider("Lama Bekerja (Tahun)", 0, 40, 5)
    jobrole = st.selectbox("Jabatan", sorted(df['JobRole'].unique()))
    dept = st.selectbox("Departemen", sorted(df['Department'].unique()))
    jobsat = st.selectbox("Kepuasan Kerja (1 = rendah, 4 = sangat puas)", [1, 2, 3, 4])

    input_data = pd.DataFrame([{
        'Age': age,
        'DistanceFromHome': distance,
        'MonthlyIncome': income,
        'OverTime': 1 if overtime == "Yes" else 0,
        'YearsAtCompany': years,
        'JobRole': jobrole,
        'Department': dept,
        'JobSatisfaction': jobsat
    }])

    data_model = df[['Age', 'DistanceFromHome', 'MonthlyIncome', 'OverTime', 'YearsAtCompany',
                     'JobRole', 'Department', 'JobSatisfaction', 'Attrition']]

    # Gabung & One-hot encode
    combined = pd.concat([data_model.drop(columns='Attrition'), input_data], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=['JobRole', 'Department'])
    input_encoded = combined_encoded.tail(1)
    X = combined_encoded.iloc[:-1]
    y = data_model['Attrition']

    # Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    pred = model.predict(input_encoded)[0]
    pred_prob = model.predict_proba(input_encoded)[0][1]

    if pred == 1:
        st.error(f"⚠️ Karyawan ini **berpotensi resign** (Probabilitas: {pred_prob:.2%})")
    else:
        st.success(f"✅ Karyawan ini **kemungkinan besar bertahan** (Probabilitas resign: {pred_prob:.2%})")

    # Alasan
    importances = model.feature_importances_
    feat_names = input_encoded.columns
    sorted_idx = importances.argsort()[::-1]

    st.markdown("### 🔍 Faktor Utama Prediksi:")
    def interpret_attr(feat):
        if "JobRole_" in feat:
            return "jenis jabatan tertentu"
        elif "Department_" in feat:
            return "jenis departemen"
        elif feat == "MonthlyIncome":
            return "gaji rendah" if income < 5000000 else "gaji tinggi"
        elif feat == "OverTime":
            return "sering lembur" if overtime == "Yes" else "tidak lembur"
        elif feat == "YearsAtCompany":
            return "lama kerja singkat" if years <= 2 else "lama kerja panjang"
        elif feat == "Age":
            return "usia muda" if age < 30 else "usia menengah/tinggi"
        elif feat == "JobSatisfaction":
            return "kepuasan kerja rendah" if jobsat <= 2 else "kepuasan kerja tinggi"
        elif feat == "DistanceFromHome":
            return "jarak rumah jauh" if distance > 20 else "jarak rumah dekat"
        else:
            return "faktor lain"

    for idx in sorted_idx[:5]:
        feat = feat_names[idx]
        score = importances[idx]

    # Ubah nama jadi lebih manusiawi
        if "JobRole_" in feat:
            readable = f"jabatan {feat.split('_')[1]}"
            alasan = "jenis jabatan tertentu"
        elif "Department_" in feat:
            readable = f"departemen {feat.split('_')[1]}"
            alasan = "jenis departemen"
        elif feat == "MonthlyIncome":
            readable = "Pendapatan Bulanan"
            alasan = "gaji rendah" if income < 5000000 else "gaji tinggi"
        elif feat == "OverTime":
            readable = "Status Lembur"
            alasan = "sering lembur" if overtime == "Yes" else "tidak lembur"
        elif feat == "YearsAtCompany":
            readable = "Lama Kerja"
            alasan = "lama kerja singkat" if years <= 2 else "lama kerja panjang"
        elif feat == "Age":
            readable = "Umur"
            alasan = "usia muda" if age < 30 else "usia menengah/tinggi"
        elif feat == "JobSatisfaction":
            readable = "Kepuasan Kerja"
            alasan = "kepuasan kerja rendah" if jobsat <= 2 else "kepuasan kerja tinggi"
        elif feat == "DistanceFromHome":
            readable = "Jarak Rumah-Kantor"
            alasan = "jarak rumah jauh" if distance > 20 else "jarak rumah dekat"
        else:
            readable = feat
            alasan = "faktor lain"

        st.markdown(f"- **{readable}** → {alasan} *(importance: {score:.2f})*")


    # Pie Chart
    st.subheader("📊 Distribusi Attrition")
    attr_count = df['Attrition'].value_counts()
    fig = px.pie(values=attr_count.values, names=['Bertahan', 'Resign'], 
                 title="Distribusi Resign Karyawan", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig)

# ============================
# 2️⃣ PREDIKSI GAJI
# ============================

elif menu == "💰 Prediksi Gaji":
    st.title("💰 Estimasi Gaji Karyawan")

    jobrole = st.selectbox("Jabatan", sorted(df['JobRole'].unique()))
    joblevel = st.selectbox("Level Jabatan", sorted(df['JobLevel'].unique()))
    education = st.selectbox("Tingkat Pendidikan", sorted(df['Education'].unique()))
    years_company = st.slider("Lama di Perusahaan (tahun)", 0, 40, 5)
    rating = st.selectbox("Performance Rating", sorted(df['PerformanceRating'].unique()))
    total_years = st.slider("Total Pengalaman Kerja", 0, 40, 6)

    input_data = pd.DataFrame([{
        'JobRole': jobrole,
        'JobLevel': joblevel,
        'Education': education,
        'YearsAtCompany': years_company,
        'PerformanceRating': rating,
        'TotalWorkingYears': total_years
    }])

    data_model = df[['JobRole', 'JobLevel', 'Education', 'YearsAtCompany',
                     'PerformanceRating', 'TotalWorkingYears', 'MonthlyIncome']]

    combined = pd.concat([data_model.drop(columns='MonthlyIncome'), input_data], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=['JobRole'])
    input_encoded = combined_encoded.tail(1)
    X = combined_encoded.iloc[:-1]
    y = data_model['MonthlyIncome']

    reg_model = RandomForestRegressor(random_state=42)
    reg_model.fit(X, y)

    salary_pred = reg_model.predict(input_encoded)[0]

    st.success(f"💵 Estimasi Gaji Bulanan: Rp {int(salary_pred):,}")

    # Alasan
    importances = reg_model.feature_importances_
    feat_names = input_encoded.columns
    sorted_idx = importances.argsort()[::-1]

    st.markdown("### 🔍 Faktor Penentu Gaji:")
    def interpret(feat):
        if "JobRole_" in feat:
            return "karena jabatan tertentu"
        elif feat == "JobLevel":
            return "karena level jabatan tinggi" if input_data['JobLevel'][0] >= 3 else "karena level jabatan rendah"
        elif feat == "TotalWorkingYears":
            return "karena pengalaman panjang" if input_data['TotalWorkingYears'][0] >= 10 else "pengalaman minim"
        elif feat == "PerformanceRating":
            return "karena performa sangat baik" if input_data['PerformanceRating'][0] == 4 else "performa standar"
        elif feat == "Education":
            return "karena pendidikan tinggi" if input_data['Education'][0] >= 4 else "pendidikan standar"
        elif feat == "YearsAtCompany":
            return "karena loyalitas lama" if input_data['YearsAtCompany'][0] >= 10 else "masih baru"
        else:
            return "kontribusi lain"

    for idx in sorted_idx[:5]:
        feat = feat_names[idx]
        score = importances[idx]

        if "JobRole_" in feat:
            readable = f"jabatan {feat.split('_')[1]}"
            alasan = "karena jabatan tertentu"
        elif feat == "JobLevel":
            readable = "Level Jabatan"
            alasan = "karena level jabatan tinggi" if input_data['JobLevel'][0] >= 3 else "karena level jabatan rendah"
        elif feat == "TotalWorkingYears":
            readable = "Total Pengalaman"
            alasan = "karena pengalaman kerja panjang" if input_data['TotalWorkingYears'][0] >= 10 else "karena pengalaman rendah"
        elif feat == "PerformanceRating":
            readable = "Rating Performa"
            alasan = "karena performa sangat baik" if input_data['PerformanceRating'][0] == 4 else "karena performa biasa"
        elif feat == "Education":
            readable = "Tingkat Pendidikan"
            alasan = "karena pendidikan tinggi" if input_data['Education'][0] >= 4 else "karena pendidikan standar"
        elif feat == "YearsAtCompany":
            readable = "Lama di Perusahaan"
            alasan = "karena loyalitas lama" if input_data['YearsAtCompany'][0] >= 10 else "karena masih cukup baru"
        else:
            readable = feat
            alasan = "kontribusi lain"

        st.markdown(f"- **{readable}** → {alasan} *(importance: {score:.2f})*")


    st.subheader("📊 Gaji vs Pengalaman")
    fig = px.scatter(df, x='TotalWorkingYears', y='MonthlyIncome', color='JobLevel',
                     title='Gaji vs Total Pengalaman Kerja')
    st.plotly_chart(fig)

# ============================
# 3️⃣ CLUSTERING
# ============================

elif menu == "👥 Segmentasi Karyawan":
    st.title("👥 Segmentasi Karyawan")

    cols = ['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)

    st.subheader("📌 Cluster: Income vs Masa Kerja")
    fig3 = px.scatter(df, x='MonthlyIncome', y='YearsAtCompany',
                      color='Cluster', hover_data=['JobRole', 'Department'],
                      title='Cluster Berdasarkan Income dan Masa Kerja')
    st.plotly_chart(fig3)

    st.subheader("📈 Rata-rata Gaji per Cluster")
    avg_income = df.groupby('Cluster')['MonthlyIncome'].mean().reset_index()
    fig4 = px.bar(avg_income, x='Cluster', y='MonthlyIncome', title='Rata-rata Gaji per Cluster')
    st.plotly_chart(fig4)

    st.subheader("📦 Usia per Cluster")
    fig5 = px.box(df, x='Cluster', y='Age', points="all", title="Distribusi Usia")
    st.plotly_chart(fig5)

    st.subheader("📊 Statistik Rata-rata")
    st.dataframe(df.groupby('Cluster')[cols].mean().round(2))
