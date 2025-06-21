import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pickle

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Load model
try:
    with open('attrition_model.pkl', 'rb') as f:
        attrition_model = pickle.load(f)
except FileNotFoundError:
    print("Error: 'attrition_model.pkl' tidak ditemukan. Jalankan 'attrition_classifier.py' terlebih dahulu.")
    exit(1)

# Load cluster data
try:
    cluster_data = pd.read_csv('cluster_data.csv')
    cluster_fig = px.scatter(cluster_data, x='PCA1', y='PCA2', color='Cluster',
                             title='Segmentasi Karyawan berdasarkan Clustering',
                             labels={'PCA1': 'Komponen PCA 1', 'PCA2': 'Komponen PCA 2'})
except FileNotFoundError:
    print("Error: 'cluster_data.csv' tidak ditemukan. Jalankan 'employee_clustering.py' terlebih dahulu.")
    cluster_fig = px.scatter()  # Placeholder jika file tidak ada

# Inisialisasi aplikasi Dash
app = dash.Dash(__name__)
app.title = 'Dasbor Analisis SDM'

# Layout halaman
app.layout = html.Div([
    html.H1("Dasbor Analisis Sumber Daya Manusia", style={'textAlign': 'center'}),
    html.Div([
        html.Label("Pilih Departemen:"),
        dcc.Dropdown(
            id='dept-filter',
            options=[{'label': dept, 'value': dept} for dept in df['Department'].unique()],
            value=None,
            placeholder='Semua Departemen',
            clearable=True
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div([
        html.H3("Prediksi Attrition"),
        html.Label("Usia:"),
        dcc.Input(id='input-age', type='number', value=30),
        html.Label("Jarak dari Rumah:"),
        dcc.Input(id='input-distance', type='number', value=10),
        html.Label("Pendidikan:"),
        dcc.Input(id='input-education', type='number', value=3),
        html.Label("Kepuasan Lingkungan:"),
        dcc.Input(id='input-env-satisfaction', type='number', value=2),
        html.Label("Kepuasan Kerja:"),
        dcc.Input(id='input-job-satisfaction', type='number', value=2),
        html.Label("Gaji Bulanan:"),
        dcc.Input(id='input-income', type='number', value=5000),
        html.Label("Lembur:"),
        dcc.Dropdown(id='input-overtime', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], value='Yes'),
        html.Label("Total Tahun Bekerja:"),
        dcc.Input(id='input-total-years', type='number', value=5),
        html.Label("Keseimbangan Kerja-Hidup:"),
        dcc.Input(id='input-worklife', type='number', value=3),
        html.Label("Tahun di Perusahaan:"),
        dcc.Input(id='input-years-company', type='number', value=2),
        html.Label("Tahun Sejak Promosi:"),
        dcc.Input(id='input-years-promotion', type='number', value=0),
        html.Label("Departemen:"),
        dcc.Dropdown(id='input-department', options=[{'label': dept, 'value': dept} for dept in df['Department'].unique()], value='Sales'),
        html.Button('Prediksi', id='predict-button', n_clicks=0),
        html.Div(id='attrition-output'),
    ], style={'width': '50%', 'margin': 'auto'}),
    dcc.Graph(id='grafik-attrition'),
    dcc.Graph(id='grafik-cluster', figure=cluster_fig)
])

# Callback untuk memperbarui grafik dan prediksi
@app.callback(
    [Output('grafik-attrition', 'figure'),
     Output('attrition-output', 'children')],
    [Input('dept-filter', 'value'),
     Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-age', 'value'),
     dash.dependencies.State('input-distance', 'value'),
     dash.dependencies.State('input-education', 'value'),
     dash.dependencies.State('input-env-satisfaction', 'value'),
     dash.dependencies.State('input-job-satisfaction', 'value'),
     dash.dependencies.State('input-income', 'value'),
     dash.dependencies.State('input-overtime', 'value'),
     dash.dependencies.State('input-total-years', 'value'),
     dash.dependencies.State('input-worklife', 'value'),
     dash.dependencies.State('input-years-company', 'value'),
     dash.dependencies.State('input-years-promotion', 'value'),
     dash.dependencies.State('input-department', 'value')]
)
def update_dashboard(dept, n_clicks, age, distance, education, env_satisfaction, job_satisfaction,
                    income, overtime, total_years, worklife, years_company, years_promotion, department):
    df_filtered = df[df['Department'] == dept] if dept else df

    # Grafik
    fig_pie = px.pie(df_filtered, names='Attrition', title='Komposisi Karyawan yang Keluar dan Bertahan')

    # Prediksi Attrition
    prediction_text = ""
    if n_clicks > 0:
        new_employee = pd.DataFrame({
            'Age': [age],
            'Department': [department],
            'DistanceFromHome': [distance],
            'Education': [education],
            'EnvironmentSatisfaction': [env_satisfaction],
            'JobSatisfaction': [job_satisfaction],
            'MonthlyIncome': [income],
            'OverTime': [overtime],
            'TotalWorkingYears': [total_years],
            'WorkLifeBalance': [worklife],
            'YearsAtCompany': [years_company],
            'YearsSinceLastPromotion': [years_promotion]
        })
        prob = attrition_model.predict_proba(new_employee)[0][1]
        prediction_text = f"Probabilitas karyawan keluar: {prob:.2f}"

    return fig_pie, prediction_text

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)