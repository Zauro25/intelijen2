import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import pickle

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Pilih fitur untuk clustering
features = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'EnvironmentSatisfaction',
            'WorkLifeBalance', 'TotalWorkingYears', 'YearsAtCompany', 'DistanceFromHome']
X = df[features]

# Preprocessing: normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tentukan jumlah cluster optimal menggunakan metode Elbow
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_curve.png')
plt.close()

# Latih model K-Means dengan jumlah cluster optimal (misalnya, k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Simpan model
with open('clustering_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Visualisasi cluster menggunakan PCA untuk reduksi dimensi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Simpan data cluster ke CSV
df[['PCA1', 'PCA2', 'Cluster']].to_csv('cluster_data.csv', index=False)

# Buat plot interaktif menggunakan Plotly (opsional)
fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                 title='Segmentasi Karyawan berdasarkan Clustering',
                 labels={'PCA1': 'Komponen PCA 1', 'PCA2': 'Komponen PCA 2'})
fig.write_html('cluster_plot.html')

# Analisis karakteristik cluster
cluster_summary = df.groupby('Cluster')[features].mean()
print("Rata-rata Fitur per Cluster:")
print(cluster_summary)