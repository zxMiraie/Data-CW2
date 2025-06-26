# Recipe Recommendation System

A simple data analytics project that finds similar recipes and predicts if you'll like them.

## What it does

This project analyzes BBC recipes and does two main things:

1. **Find Similar Recipes**: Give it a recipe you like, and it finds similar ones
2. **Predict Taste**: Tells you if a recipe will be "tasty" (rating > 4.2)

## How to use

1. Install required packages:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn tensorflow
```

2. Run the script:

```bash
python main.py
```

## What's inside

- `recipes.csv` - BBC recipes data with ratings, ingredients, and cooking info
- `main.py` - The main script that does everything

## How it works

The system uses three different methods to find similar recipes:

- **Cosine Similarity**: Compares recipe features mathematically
- **Vector Space Model**: Uses text analysis to find similarities
- **K-Nearest Neighbors**: Finds the closest matching recipes

For taste prediction, it uses Logistic Regression on recipes with enough ratings (13+).
