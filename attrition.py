import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# Load data
df = pd.read_csv('HR_Analytics.csv')

# Pilih fitur dan target
features = ['Age', 'DistanceFromHome', 'JobSatisfaction', 'EnvironmentSatisfaction',
            'MonthlyIncome', 'YearsAtCompany', 'OverTime', 'JobRole', 'Department']
target = 'Attrition'

# Encode kolom kategorik
le = LabelEncoder()
df['OverTime'] = le.fit_transform(df['OverTime'])
df['JobRole'] = le.fit_transform(df['JobRole'])
df['Department'] = le.fit_transform(df['Department'])
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
st.subheader("📊 Distribusi Karyawan Resign vs Tidak")
attrition_counts = df['Attrition'].value_counts().rename({0: "Tidak Resign", 1: "Resign"})
fig = px.pie(names=attrition_counts.index, values=attrition_counts.values,
             title="Distribusi Karyawan", color_discrete_sequence=px.colors.sequential.RdBu)
st.plotly_chart(fig)
