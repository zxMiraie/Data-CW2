import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Imports the recipes.csv into a Pandas dataframe
df = pd.read_csv('recipes.csv')

#Treats values that ONLY have empty spaces
df['cuisine'] = df['cuisine'].replace(r'^ $', 'Unknown', regex=True)
df['category'] = df['category'].replace(r'^ $', 'Unknown', regex=True)

#Give Summary
print(df.describe())

top_rated = df.nlargest(10, 'rating_avg')
print(top_rated)