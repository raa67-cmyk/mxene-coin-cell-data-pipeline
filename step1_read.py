import pandas as pd

df = pd.read_csv("data/raw/cell01.csv", parse_dates=["timestamp"])
print(df.dtypes)       # show data types
print(df.head(4))      # first 4 rows
print(df.tail(4))      # last 4 rows
