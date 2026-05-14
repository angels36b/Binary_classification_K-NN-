import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('adult_data_train.csv')

#delate columns education - marital
df_cleaned = df.drop(columns=['education', 'marital-status'])
print(df_cleaned.head())

#we get the types of all the collumns/ получаем типы всех столбцов
data_types = df_cleaned.dtypes
#we count how many are type object and numeric / подсчитаем, склоько из них имеют тип object
non_numeric_count = (data_types == 'object').sum()

numeric_count = (data_types == 'int64').sum() + (data_types == 'float64').sum()

print(f"object type:{non_numeric_count}")
print(f"numeric type:{numeric_count}")