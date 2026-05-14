import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('adult_data_train.csv')

#delate columns education - marital
df_cleaned = df.drop(columns=['education', 'marital-status'])
print(df_cleaned.head())
