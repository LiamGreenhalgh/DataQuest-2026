import pandas as pd

df1 = pd.read_csv('input.csv')
df1['Mobility'] = df1['Mobility'] / df1['Retention Type']
df1 = df1.drop(columns='Retention Type')
df1.to_csv('output.csv', index=False)