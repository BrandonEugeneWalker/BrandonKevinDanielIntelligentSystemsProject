#/usr/local/bin/python3.7.0
"""
Processes data from the steam store and then uses it to learn with Linear Regression, Decision Trees, and Random Forests. 
"""

import numpy as np
import pandas as pd
import csv
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def steam_file_processor(file_name):
    """
    Cleans and preprocesses the given csv file.
    Cleaning involves removing unused columns and converting columns we want to use into relevant data.
    Expects a file following the format of the Steam Store Dataset (clean)'s steam.csv.

    Used Columns: 
        positive_ratings, negative_ratings, owners, average_playtime, median_playtime, and price

    Unused Columns: 
        appid, name, release_date, english, developer, publisher, platforms, required_age, categories, genres, steamspy_tags, achievements

    Columns that need cleaning: 
        owners
            Owners is a range between two numbers, therefore for us to be able to use it we need to transform the data into a singular
            value that is easy to understand and use. This value will simply be the average of the two range values.

    Why certain columns were removed:
        appid
            There is no correlation between the appid (a id used by steam that is not seen by the user) and the price of a game.
            Including this column would simply create noise for what we want to predict.
        name

        release_date

        english

        developer

        publisher

        platforms

        required_age

        categories

        genres

        steamspy_tags

        acheivements

    """
    df = pd.read_csv(file_name)
    df['positive_ratings_'] = df['positive_ratings'].astype(int)
    df['negative_ratings_'] = df['negative_ratings'].astype(int)
    df['owners_'] = df['owners'].astype(int) #might need to change, as it is a range, not a specific number
    df['average_playtime_'] = df['average_playtime'].astype(int)
    df['median_playtime_'] = df['median_playtime'].astype(int)
    df['price_'] = df['price'].astype(float)
    #set our x and y
    tree_regressor = DecisionTreeRegressor(criterion='mse') # taking Mean Square Error
    
def steam_data_cleaner():
    min_owner = 0
    max_owner = 1
    df = pd.read_csv("steam.csv")
    steam_row_length = len(df.index)
    print(steam_row_length)
    for i in range(0, steam_row_length):
        owner_range = df["owners"][i]
        owner_range_array = owner_range.split("-")
        min_owner_value = int(owner_range_array[min_owner])
        max_owner_value = int(owner_range_array[max_owner])
        owner_avg_value = (min_owner_value + max_owner_value) / len(owner_range_array)
        df["owners"][i] = owner_avg_value
    df.to_csv("steam_cleaned.csv")


def steam_learning_regression(data):
    """
    Trains a multiple linear regression model using the given data.
    The trained model is returned.
    """

def steam_learning_tree(data):
    """
    Trains a decision tree model using the given data.
    The trained model is returned.
    """

def steam_learning_forest(data):
    """
    Trains a random forest model using the given data.
    The trained model is returned.
    """


steam_data_cleaner()