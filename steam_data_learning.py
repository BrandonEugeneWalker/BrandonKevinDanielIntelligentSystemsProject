#/usr/local/bin/python3.7.0
"""
Processes data from the steam store and then uses it to learn with Linear Regression, Decision Trees, and Random Forests. 
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def steam_file_processor(file_name):
    """

    """
df = pd.read_csv('steam.csv')
df['positive_ratings_'] = df['positive_ratings'].astype(int)
df['negative_ratings_'] = df['negative_ratings'].astype(int)
df['owners_'] = df['owners'].astype(int) #might need to change, as it is a range, not a specific number
df['average_playtime_'] = df['average_playtime'].astype(int)
df['median_playtime_'] = df['median_playtime'].astype(int)
df['price_'] = df['price'].astype(float)

def steam_learning_regression(data):
    """

    """

def steam_learning_tree(data):
    """

    """

def steam_learning_forest(data):
    """

    """

