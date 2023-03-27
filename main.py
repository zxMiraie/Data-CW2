import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Imports the recipes.csv into a Pandas dataframe
df = pd.read_csv('recipes.csv')
features = ['title', 'rating_avg', 'rating_val', 'total_time', 'category', 'cuisine', 'ingredients']

# Treats values that ONLY have empty spaces
df['cuisine'] = df['cuisine'].replace(r'^ $', 'Unknown', regex=True)
df['category'] = df['category'].replace(r'^ $', 'Unknown', regex=True)

# # summary statistics
# print(df.describe())
#
# top_rated = df.nlargest(10, 'rating_avg')
# print(top_rated)

# # Visualize average ratings and number of ratings
# plt.figure(figsize=(12, 6))
# sns.scatterplot(data=df, x='rating_avg', y='rating_val')
# plt.xlabel('Average Rating')
# plt.ylabel('Number of Ratings')
# plt.show()

# Calculate the percentile or mean of the number of ratings
threshold = df['rating_val'].quantile(0.75)  # 75th percentile

print(f'Threshold for number of ratings: {threshold}')

#####


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


def get_recommendations(title, cosine_sim_matrix, data, top_n=10):
    try:
        index = df[df["title"] == title].index[0]
        similarity_scores = list(enumerate(cosine_sim_matrix[index]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similarity_scores = similarity_scores[1:top_n + 1]

        indices = [i[0] for i in similarity_scores]
        return data.iloc[indices]["title"]
    except IndexError:
        return "Inputted title is not in dataframe"


# liked_recipe = "Chicken and coconut curry"
# recommendations = get_recommendations(liked_recipe, cosine_sim, df)
# print(recommendations)


########################################################

def vec_space_method(recipe, df):
    try:
        # Fit and transform the combine_features column into a matrix of token counts
        count_matrix = vectorizer.fit_transform(df['combine_features'])

        # Compute the pairwise cosine similarity between the given recipe and each row in the count matrix
        recipe_vec = vectorizer.transform([recipe])
        similarity_scores = cosine_similarity(recipe_vec, count_matrix)[0]

        # Get the indices of the top 10 most similar recipes
        top_indices = similarity_scores.argsort()[::-1][:10]

        # Return the titles of the top 10 most similar recipes
        return df.iloc[top_indices]['title']
    except IndexError:
        print("Inputted title is not in dataframe")


recipe_title = "Chicken and coconut curry"
similar_recipes = vec_space_method(recipe_title, df)
print(similar_recipes)


def knn_similarity(recipe_title, top_n=10):
    # Fit and transform the combine_features column into a matrix of token counts
    count_matrix = vectorizer.fit_transform(df['combine_features'])

    # Calculate the pairwise euclidean distances between each pair of rows in the count matrix
    euclidean_dist = euclidean_distances(count_matrix, count_matrix)

    # Fit the nearest neighbors model to the euclidean distances
    model = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine', algorithm="auto").fit(euclidean_dist)

    # Find the index of the input recipe in the dataframe
    recipe_index = df.index[df['title'] == recipe_title].tolist()[0]

    # Find the indices of the 10 most similar recipes using the KNN algorithm
    distances, indices = model.kneighbors([euclidean_dist[recipe_index]], n_neighbors=top_n + 1)

    # Get the titles of the recommended recipes
    recommended_recipes = df.iloc[indices[0][1:]]['title']

    return recommended_recipes


recipe_title = "Chicken and coconut curry"
similar_recipes = knn_similarity(recipe_title)
print(similar_recipes)

# Task 6
test_set = [
    "Chicken tikka masala",
    "Albanian baked lamb with rice (Tavë kosi)",
    "Baked salmon with chorizo rice",
    "Almond lentil stew"]

test_df = pd.DataFrame()
vec_space_test = pd.DataFrame()
knn_test = pd.DataFrame()

for i in range(len(test_set)):
    # Running the test_set through both recommenders and adding the results into sets
    similar_recipes = vec_space_method(test_set[i], df)
    vec_space_test = pd.concat([vec_space_test, similar_recipes], axis=0)
    similar_recipes = knn_similarity(test_set[i])
    knn_test = pd.concat([knn_test, similar_recipes], axis=0)


def coverage(input_set, dataset):
    # Counts the unique number of items in the set and then divides it by the total number of recipes
    input_set = input_set[0].nunique()
    return input_set / len(dataset)


def personalisation(input_set):
    # Transforms the set into a cosine matrix to calculate the personalisation
    input_set_matrix = vectorizer.fit_transform(input_set[0])
    input_set_matrix_cos = cosine_similarity(input_set_matrix)
    return 1 - np.mean(input_set_matrix_cos)


print(f'Vector Space Method coverage = {coverage(vec_space_test, df)}')
print(f'KNN coverage = {coverage(knn_test, df)}')

print(f'Vector Space Method personalisation = {personalisation(vec_space_test)}')
print(f'KNN personalisation = {personalisation(knn_test)}')

# Task 7

df["tasty"] = df["rating_avg"].apply(lambda x: 1 if x > 4.2 else -1)

X = df.iloc[:,6]
Y = df.iloc[:,12]

model = Sequential()
model.add(Dense(12, activation="relu"))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, Y ,epochs=150, batch_size=10, verbose=1)

loss, accuracy = model.evaluate(X, Y, verbose=1)
print("Accuracy: %.2f%%" % (accuracy*100))
print('Test loss:', loss)