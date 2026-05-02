import pandas as pd

df1 = pd.read_csv('input.csv')
df1 = df1.dropna(subset=['PctProficient'])
df1.to_csv('output.csv', index=False)