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

# Visualize average ratings and number of ratings
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='rating_avg', y='rating_val')
plt.xlabel('Average Rating')
plt.ylabel('Number of Ratings')
plt.show()

# Calculate the percentile or mean of the number of ratings
threshold = df['rating_val'].quantile(0.75)  # 75th percentile
# Alternatively, you can use the mean:
# threshold = data['rating_val'].mean()

print(f'Threshold for number of ratings: {threshold}')

features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']

# Define a lambda function to combine the features in each row
combine_func = lambda row: ' '.join([str(row[feature]) for feature in features])

# Apply the lambda function to each row of the DataFrame to create a new column
df['combine_features'] = df.apply(combine_func, axis=1)

# # Display the updated DataFrame
# print(df.head())
