import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('recipes.csv')

# Identify missing values
missing_values = df.isnull().sum()

# Replace missing values with mean of the column
df = df.fillna(df.mean())

# Summary statistics
print(df.describe())

# 10 highest rated recipes
top_rated = df.sort_values('rating', ascending=False).head(10)
print(top_rated)