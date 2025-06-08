import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output

# Load data
df = pd.read_csv('HR_Analytics.csv')

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

    dcc.Graph(id='grafik-attrition'),
    dcc.Graph(id='grafik-usia-gaji'),
    dcc.Graph(id='grafik-kepuasan'),
])

# Callback untuk memperbarui grafik
@app.callback(
    [Output('grafik-attrition', 'figure'),
     Output('grafik-usia-gaji', 'figure'),
     Output('grafik-kepuasan', 'figure')],
    [Input('dept-filter', 'value')]
)
def update_graphs(departemen_terpilih):
    df_filtered = df[df['Department'] == departemen_terpilih] if departemen_terpilih else df

    # Pie chart: attrition
    fig_pie = px.pie(df_filtered, names='Attrition', title='Komposisi Karyawan yang Keluar dan Bertahan')

    # Scatter: usia vs gaji
    fig_scatter = px.scatter(df_filtered, x='Age', y='MonthlyIncome', color='Attrition',
                             title='Sebaran Usia dan Gaji Bulanan Karyawan',
                             labels={'Age': 'Usia', 'MonthlyIncome': 'Gaji Bulanan'})

    # Histogram: kepuasan kerja
    fig_bar = px.histogram(df_filtered, x='JobSatisfaction', nbins=4,
                           title='Distribusi Kepuasan Kerja',
                           labels={'JobSatisfaction': 'Tingkat Kepuasan Kerja'})

    return fig_pie, fig_scatter, fig_bar

# Jalankan server
if __name__ == '__main__':
    app.run(debug=True)
