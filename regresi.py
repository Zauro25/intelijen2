import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import streamlit as st
import joblib


# Load data
df = pd.read_csv('HR_Analytics.csv')

# Encode kolom kategorik
le_jobrole = LabelEncoder()
df['JobRole'] = le_jobrole.fit_transform(df['JobRole'])

# Fitur dan target
features = ['JobLevel', 'Education', 'YearsAtCompany',
            'PerformanceRating', 'TotalWorkingYears', 'JobRole']
target = 'MonthlyIncome'

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
st.subheader("📊 Distribusi Gaji terhadap Pengalaman")
fig2 = px.scatter(df, x='TotalWorkingYears', y='MonthlyIncome', color='JobLevel',
                  title='Gaji Bulanan vs Total Pengalaman Kerja',
                  labels={'TotalWorkingYears': 'Total Working Years', 'MonthlyIncome': 'Monthly Income'})
st.plotly_chart(fig2)
joblib.dump(model, 'model_regresi.pkl')
joblib.dump(le_jobrole, 'encoder_jobrole.pkl')