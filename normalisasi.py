import pandas as pd

df = pd.read_csv("HR_Analytics.csv")
print(df['MonthlyIncome'].describe())
print(df['MonthlyIncome'].value_counts())
