#/usr/local/bin/python3.7.0
"""
Processes data from the steam store and then uses it to learn with Linear Regression, Decision Trees, and Random Forests. 
"""

import numpy as np
import pandas as pd
import csv
from datetime import datetime
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from numpy import mean

def steam_file_processor(file_name):
    df = pd.read_csv(file_name)
    df['positive_ratings_'] = df['positive_ratings'].astype(int)
    df['negative_ratings_'] = df['negative_ratings'].astype(int)
    df['owners_'] = df['owners'].astype(float)
    df['average_playtime_'] = df['average_playtime'].astype(int)
    df['median_playtime_'] = df['median_playtime'].astype(int)
    df['price_'] = df['price'].astype(float)
    
    return df
    
def steam_data_cleaner(file_name):
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
            While a name can increase marketability and potential price of a game, a much larger machine learning program would be 
            required to create a training dataset based on game names and their price and marketability.
        release_date
            Insufficient data and record of a game's price over time to predict price correlating to its initial release date.
        english
            A binary value to track if a game is released in English, does not correlate to the price of a game as internationally, 
            English is a minority of languages and one of many spoken and used in games.
        developer
            Like with names, no predictable correlation between a game's developer and the price without utilizing a larger and more 
            complicated algorithm. Resulting in a noise column.
        publisher
            A publisher can have influence on a games market share and value, but has questionable correlation and also goes into the depth
            of larger and more complicated programs.
        platforms
            Platforms have no influence on the game's price, just its availability of operating system. A noise column.
        required_age
            No direct correlation to the game's price, games of all prices available in each age range / requirement.
        categories
            No influence on price based on its categories. A noise column.
        genres
            No direct influence on price. A noise column.
        steamspy_tags
            A noise column due to the tags having no direct influence on price. Noise column.
        achievements
            A game containing achievements does not affect the game's price. Noise column.
    """
    min_owner = 0
    max_owner = 1
    df = pd.read_csv(file_name)
    steam_row_length = len(df.index)
    for i in range(0, steam_row_length):
        owner_range = df["owners"][i]
        owner_range_array = owner_range.split("-")
        min_owner_value = int(owner_range_array[min_owner])
        max_owner_value = int(owner_range_array[max_owner])
        owner_avg_value = (min_owner_value + max_owner_value) / len(owner_range_array)
        df["owners"][i] = owner_avg_value
    df.to_csv("steam_cleaned.csv", columns=["positive_ratings", "negative_ratings", "owners", "average_playtime", "median_playtime", "price"], index=False)

def steam_learning_regression(data):
    """
    Trains a multiple linear regression model using the given data.
    The trained model is returned.
    """
    NUM_FOLDS = 10
    regression_train = data[["positive_ratings", "negative_ratings", "owners", "average_playtime", "median_playtime", "price"]]
    regression_label = data[["price"]]
    regression_model = linear_model.LinearRegression()
    regression_model.fit(regression_train, regression_label)

    linear_classifier = linear_model.HuberRegressor()
    skf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)

    fold = 0
    overall_mse = []
    for train_index, test_index in skf.split(regression_train, regression_label):
        x_train_fold = [df.loc[i] for i in train_index]
        y_train_fold = [df.loc[i] for i in train_index]
        x_test_fold = [df.loc[i] for i in test_index]
        y_test_fold = [df.loc[i] for i in test_index]

        linear_classifier.fit(x_train_fold, y_train_fold)
        preds = linear_classifier.predict(x_test_fold)
        mse = metrics.mean_squared_error(y_test_fold, preds)
        print("fold", fold, "#train:", len(train_index), "#test:", len(preds), "total:", (len(train_index) + len(preds)), "MSE:", mse)

        overall_mse.append(mse)
        fold+= 1
    final_results = str("Mean MSE over", NUM_FOLDS, "folds:", mean(overall_mse))
    print(final_results)
    return final_results


def steam_learning_tree(data):
    """
    Trains a decision tree model using the given data.
    The trained model is returned.
    K-Fold fold amount was decided by running the model with diferent amounts of folds.
    Taking roughly 32 seconds per fold we decided that 10 folds would be enough.
    This is because while it only takes ~5 minutes for the tree folds to run it takes longer for the forest.
    """
    NUM_FOLDS = 10
    tree_train = data[["positive_ratings", "negative_ratings", "owners", "average_playtime", "median_playtime", "price"]]
    tree_label = data[["price"]]
    tree_classifier = DecisionTreeRegressor(criterion="mse")
    skf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)

    fold = 0
    overall_mse = []
    for train_index, test_index in skf.split(tree_train, tree_label):
        x_train_fold = [df.loc[i] for i in train_index]
        y_train_fold = [df.loc[i] for i in train_index]
        x_test_fold = [df.loc[i] for i in test_index]
        y_test_fold = [df.loc[i] for i in test_index]

        tree_classifier.fit(x_train_fold, y_train_fold)
        preds = tree_classifier.predict(x_test_fold)
        mse = metrics.mean_squared_error(y_test_fold, preds)
        print("fold", fold, "#train:", len(train_index), "#test:", len(preds), "total:", (len(train_index) + len(preds)), "MSE:", mse)

        overall_mse.append(mse)
        fold+= 1
    final_results = str("Mean MSE over", NUM_FOLDS, "folds:", mean(overall_mse))
    print(final_results)
    return final_results

def steam_learning_forest(data):
    """
    Trains a random forest model using the given data.
    The trained model is returned.
    Number of trees was measured for time efficiency after the rate of decrease in the error diminished. 
    At ~200, this peaks. If we choose arbitrarily larger, 1500 trees, we only achieve a decrease in the thousandths.
    """
    trees = 200
    NUM_FOLDS = 10

    X = data.iloc[:, 0:5].values
    y = data.iloc[:, 5].values
    skf = StratifiedKFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)
    regressor = RandomForestRegressor(n_estimators=trees, random_state=0)

    fold = 0
    overall_mse = []
    for train_index, test_index in skf.split(X, y):

        x_train_fold = [df.loc[i] for i in train_index]
        y_train_fold = [df.loc[i] for i in train_index]
        x_test_fold = [df.loc[i] for i in test_index]
        y_test_fold = [df.loc[i] for i in test_index]

        regressor.fit(x_train_fold, y_train_fold)
        preds = regressor.predict(x_test_fold)
        mse = metrics.mean_squared_error(y_test_fold, preds)
        print("fold", fold, "#train:", len(train_index), "#test:", len(preds), "total:", (len(train_index) + len(preds)), "MSE:", mse)

        overall_mse.append(mse)
        fold += 1
    
    final_results = str("Mean MSE over", NUM_FOLDS, "folds:", mean(overall_mse))
    print(final_results)
    return final_results


starting_csv = "steam.csv"
clean_csv = "steam_cleaned.csv"
df = steam_file_processor(clean_csv)

#Running and timing Regression
regression_start = datetime.now()
regression_results = steam_learning_regression(df)
regression_end = datetime.now()
regression_total_time = regression_end - regression_start
print('Regression Total Time: ', regression_total_time)

#Running and timing Decision Tree
tree_start = datetime.now()
tree_results = steam_learning_tree(df)
tree_end = datetime.now()
tree_total_time = tree_end - tree_start
print('Decision Tree Total Time: ', tree_total_time)

#Running and timing Random Forest
forest_start = datetime.now()
forest_results = steam_learning_forest(df)
forest_end = datetime.now()
forest_total_time = forest_end - forest_start
print('Random Forest Total Time: ', forest_total_time)

#Printing results again.
print("---Linear Regression---")
print(regression_results)
print("---Tree Regression---")
print(tree_results)
print("---Random Forest---")
print(forest_results)