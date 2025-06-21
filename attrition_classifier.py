import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Pilih fitur dan target
features = ['Age', 'Department', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'JobSatisfaction', 'MonthlyIncome', 'OverTime', 'TotalWorkingYears',
            'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion']
X = df[features]
y = df['Attrition']

# Preprocessing: pisahkan kolom numerik dan kategorikal
numeric_features = ['Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                    'JobSatisfaction', 'MonthlyIncome', 'TotalWorkingYears',
                    'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion']
categorical_features = ['Department', 'OverTime']

# Buat pipeline preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Buat pipeline model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Simpan model
with open('attrition_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Contoh prediksi untuk karyawan baru
new_employee = pd.DataFrame({
    'Age': [30],
    'Department': ['Sales'],
    'DistanceFromHome': [10],
    'Education': [3],
    'EnvironmentSatisfaction': [2],
    'JobSatisfaction': [2],
    'MonthlyIncome': [5000],
    'OverTime': ['Yes'],
    'TotalWorkingYears': [5],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [2],
    'YearsSinceLastPromotion': [0]
})
prediction = model.predict_proba(new_employee)
print(f"Probabilitas Attrition (Yes): {prediction[0][1]:.2f}")