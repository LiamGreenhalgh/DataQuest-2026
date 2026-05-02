import pandas as pd

df1 = pd.read_csv('input.csv')
df2 = df1[['Performance', 'Mobility']]
df2.to_csv('output.csv', index=False)

