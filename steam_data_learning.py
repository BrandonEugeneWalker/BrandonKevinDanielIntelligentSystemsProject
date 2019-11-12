#/usr/local/bin/python3.7.0
"""
Processes data from the steam store and then uses it to learn with Linear Regression, Decision Trees, and Random Forests. 
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

def steam_file_processor(file_name):
    """

    """
df = pd.read_csv('steam.csv')
lb = LabelEncoder()
df['positive_ratings_'] = lb.fit_transform(df['positive_ratings'])
df['negative_ratings_'] = lb.fit_transform(df['negative_ratings'])
df['owners_'] = lb.fit_transform(df['owners'])
df['average_playtime_'] = lb.fit_transform(df['average_playtime'])
df['median_playtime_'] = lb.fit_transform(df['median_playtime'])
df['price_'] = lb.fit_transform(df['price'])

def steam_learning_regression(data):
    """

    """

def steam_learning_tree(data):
    """

    """

def steam_learning_forest(data):
    """

    """

