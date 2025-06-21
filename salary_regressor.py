import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Pilih fitur dan target
features = ['Age', 'Department', 'Education', 'EducationField', 'JobLevel',
            'JobRole', 'TotalWorkingYears', 'YearsAtCompany', 'PerformanceRating',
            'YearsSinceLastPromotion']
X = df[features]
y = df['MonthlyIncome']

# Preprocessing: pisahkan kolom numerik dan kategorikal
numeric_features = ['Age', 'Education', 'TotalWorkingYears', 'YearsAtCompany',
                    'PerformanceRating', 'YearsSinceLastPromotion']
categorical_features = ['Department', 'EducationField', 'JobRole']

# Buat pipeline preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Buat pipeline model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
])

# Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Simpan model
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Contoh prediksi untuk karyawan baru
new_employee = pd.DataFrame({
    'Age': [35],
    'Department': ['Research & Development'],
    'Education': [4],
    'EducationField': ['Life Sciences'],
    'JobLevel': [3],
    'JobRole': ['Manager'],
    'TotalWorkingYears': [10],
    'YearsAtCompany': [5],
    'PerformanceRating': [3],
    'YearsSinceLastPromotion': [2]
})
prediction = model.predict(new_employee)
print(f"Prediksi Gaji Bulanan: ${prediction[0]:.2f}")