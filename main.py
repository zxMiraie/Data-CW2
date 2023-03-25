import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors



# Imports the recipes.csv into a Pandas dataframe
df = pd.read_csv('recipes.csv')
features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']


#Treats values that ONLY have empty spaces
df['cuisine'] = df['cuisine'].replace(r'^ $', 'Unknown', regex=True)
df['category'] = df['category'].replace(r'^ $', 'Unknown', regex=True)


#summary statistics
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


print(f'Threshold for number of ratings: {threshold}')


# Define a lambda function to combine the features in each row
combine_func = lambda row: ' '.join([str(row[feature]) for feature in features])

# Apply the lambda function to each row of the DataFrame to create a new column
df['combine_features'] = df.apply(combine_func, axis=1)

# Display the updated DataFrame
print(df.head())

vectorizer = CountVectorizer()

# Fit and transform the combine_features column into a matrix of token counts
count_matrix = vectorizer.fit_transform(df['combine_features'])

# Compute the pairwise cosine similarity between each pair of rows in the count matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Display the cosine similarity matrix
print(cosine_sim)


# def get_recommendations(title, cosine_sim_matrix, data, top_n=10):
#     try:
#         index = df[df["title"] == title].index[0]
#         similarity_scores = list(enumerate(cosine_sim_matrix[index]))
#         similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#         similarity_scores = similarity_scores[1:top_n + 1]
#
#         indices = [i[0] for i in similarity_scores]
#         return data.iloc[indices]["title"]
#     except IndexError:
#         return "Inputted title is not in dataframe"
#
#
# liked_recipe = "Chicken and coconut curry"
# recommendations = get_recommendations(liked_recipe, cosine_sim, df)
# print(recommendations)


#########################################################

# def custom_distance_metric(x, y):
#     categorical_features = ["category", "cuisine"]
#     numerical_features = ["rating_avg", "total_time"]
#
#     cat_dist = sum([1 if x[feature] != y[feature] else 0 for feature in categorical_features])
#     num_dist = euclidean_distances([x[numerical_features].values], [y[numerical_features].values])[0][0]
#
#     return cat_dist + num_dist
#
#
# def vec_space_method(recipe_title, data, top_n=10):
#     selected_recipe = data[data["title"] == recipe_title].iloc[0]
#     data["distance"] = data.apply(lambda x: custom_distance_metric(selected_recipe, x), axis=1)
#     closest_recipes = data.sort_values("distance").head(top_n + 1).iloc[1:]["title"]
#     return closest_recipes
#
#
# recipe_title = "Chicken and coconut curry"
# similar_recipes = vec_space_method(recipe_title, df)
# print(similar_recipes)


def knn_similarity(recipe_title, data, top_n=10):
    categorical_data = pd.get_dummies(data[["category", "cuisine"]])
    numerical_data = data[["rating_avg", "total_time"]]

    scaler = MinMaxScaler()
    numerical_data_scaled = scaler.fit_transform(numerical_data)

    features = pd.concat(
        [pd.DataFrame(numerical_data_scaled, columns=["rating_avg", "total_time"]), categorical_data], axis=1)

    model = NearestNeighbors(n_neighbors=top_n + 1, algorithm="auto", metric="euclidean").fit(features)
    index = data[data["title"] == recipe_title].index[0]

    distances, indices = model.kneighbors(features.iloc[index].values.reshape(1, -1))
    return data.iloc[indices[0][1:]]["title"]

recipe_title = "Chicken and coconut curry"
similar_recipes = knn_similarity(recipe_title, df)
print(similar_recipes)