from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Pilih kolom untuk clustering
cols = ['Age', 'YearsAtCompany', 'MonthlyIncome', 'JobSatisfaction', 'PerformanceRating']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[cols])

# Latih KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualisasi hasil clustering
plt.figure(figsize=(8, 5))
plt.scatter(df['MonthlyIncome'], df['YearsAtCompany'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Monthly Income')
plt.ylabel('Years At Company')
plt.title('Clustering Karyawan Berdasarkan Income & Masa Kerja')
plt.colorbar(label='Cluster')
plt.show()
st.subheader("📈 Rata-rata Gaji per Cluster")
avg_income = df.groupby('Cluster')['MonthlyIncome'].mean().reset_index()
fig3 = px.bar(avg_income, x='Cluster', y='MonthlyIncome',
              title='Rata-rata Gaji per Cluster',
              labels={'MonthlyIncome': 'Monthly Income'})
st.plotly_chart(fig3)
st.subheader("📦 Penyebaran Usia per Cluster")
fig4 = px.box(df, x='Cluster', y='Age', points="all", title="Distribusi Usia di Tiap Cluster")
st.plotly_chart(fig4)


