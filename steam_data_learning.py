#/usr/local/bin/python3.7.0
"""
Cleans and processes data from the steam store and then uses it to learn with Linear Regression, Decision Trees, and Random Forests with k-fold validation.

K-Fold fold amount was decided by running the model with diferent amounts of folds.
Taking roughly 32 seconds per fold (depends on the computer) we decided that 10 folds would be enough.
This is because while it only takes ~5 minutes for the tree folds to run it takes longer for the forest.
It takes roughly half an hour to run the entire script.
"""

import numpy as np
from numpy import mean
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor

def steam_file_processor(file_name):
    """
    Opens the given file and reads it into a pandas dataframe.
    The data is then set as its initial value types and returned.
    Expects the cleaned dataset csv file.
    """
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
    The final results are saved as a new csv file named steam_cleaned.csv.

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

def steam_learning_regression(data, NUM_FOLDS):
    """
    Trains a multiple linear regression model using the given data.
    Uses K-Fold validation with NUM_FOLDS folds.
    A string describing the results is retuned.
    Takes roughly 8 minutes to run.
    """
    regression_train = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    regression_label = data[["price_"]]
    regression_model = linear_model.LinearRegression()

    skf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)

    mse_scorer = make_scorer(mean_squared_error)
    results = cross_val_score(regression_model, regression_train, regression_label, scoring=mse_scorer, cv=skf)
    print(f"Boosting - MSE Array: {results}")

    mean_overall = np.mean(results)
    final_results = f"Regression - Mean MSE over {NUM_FOLDS} folds: {mean_overall}"
    print(final_results)
    return final_results


def steam_learning_tree(data, NUM_FOLDS):
    """
    Trains a decision tree model using the given data.
    Uses K-Fold validation with NUM_FOLDS folds.
    A string describing the results is returned.
    Takes roughly 8 minutes to run.
    """
    tree_train = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    tree_label = data[["price_"]]
    tree_regression = DecisionTreeRegressor(criterion="mse")
    skf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)

    mse_scorer = make_scorer(mean_squared_error)
    results = cross_val_score(tree_regression, tree_train, tree_label, scoring=mse_scorer, cv=skf)
    print(f"Boosting - MSE Array: {results}")

    mean_overall = np.mean(results)
    final_results = f"Regression - Mean MSE over {NUM_FOLDS} folds: {mean_overall}"
    print(final_results)
    return final_results

def steam_learning_forest(data, NUM_FOLDS):
    """
    Trains a random forest model using the given data.
    Uses K-Fold validation with NUM_FOLDS folds.
    A string describing the results is returned.
    Takes roughly 8 minutes to run.
    Number of trees was measured for time efficiency after the rate of decrease in the error diminished. 
    At ~200, this peaks. If we choose arbitrarily larger, 1500 trees, we only achieve a decrease in the thousandths.
    """
    trees = 200

    forest_train = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    forest_label = data[["price_"]]
    
    skf = KFold(n_splits=NUM_FOLDS, random_state=None, shuffle=True)
    forest_regressor = RandomForestRegressor(n_estimators=trees, random_state=0)

    mse_scorer = make_scorer(mean_squared_error)
    results = cross_val_score(forest_regressor, forest_train, forest_label.values.ravel(), scoring=mse_scorer, cv=skf)
    print(f"Boosting - MSE Array: {results}")

    mean_overall = np.mean(results)
    final_results = f"Regression - Mean MSE over {NUM_FOLDS} folds: {mean_overall}"
    print(final_results)
    return final_results

def steam_learning_bagging(data, NUM_FOLDS):
    """
    Ensemble BaggingRegressor using DecisionRegressor
    Uses K-Fold validation with NUM_FOLDS folds.
    A string describing the results is returned.
    Number of trees was measured for time efficiency after the rate of decrease in the error diminished. 
    At ~200, this peaks. If we choose arbitrarily larger, 1500 trees, we only achieve a decrease in the thousandths.
    Seed set for predictable results
    """
    trees = 200
    seed = 7

    X = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    y = data[["price_"]]

    kfold = KFold(n_splits=NUM_FOLDS, random_state=seed)
    base_cls = DecisionTreeRegressor()

    model = BaggingRegressor(base_estimator=base_cls, n_estimators=trees, random_state=seed)
    mse_scorer = make_scorer(mean_squared_error)
    results = cross_val_score(model, X, y.values.ravel(), scoring=mse_scorer, error_score='raise', cv=kfold)
    print(f"Bagging - MSE Array: {results}")

    final_results = f"Bagging - Mean MSE over {NUM_FOLDS} folds: {np.mean(results)}"
    print(final_results)
    return(final_results)

def steam_learning_boosting(data, NUM_FOLDS):
    """
    Ensemble BoostingRegressor to boost over each fold
    Uses K-Fold validation with NUM_FOLDS folds.
    A string describing the results is returned.
    Seed set for predictable results
    """
    seed = 7

    X = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    y = data[["price_"]]

    kfold = KFold(n_splits=NUM_FOLDS, random_state=seed)
    
    model = GradientBoostingRegressor()
    mse_scorer = make_scorer(mean_squared_error)
    results = cross_val_score(model, X, y.values.ravel(), scoring=mse_scorer, cv=kfold)
    print(f"Boosting - MSE Array: {results}")

    final_results = f"Boosting - Mean MSE over {NUM_FOLDS} folds: {np.mean(results)}"
    print(final_results)
    return(final_results)

def steam_learning_voting(data, NUM_FOLDS):
    """
    Voting regressor that combines different types of regressors to try and overcome their weaknesses.
    """
    X = data[["positive_ratings_", "negative_ratings_", "owners_", "average_playtime_", "median_playtime_"]]
    y = data[["price_"]]


    kfold = KFold(n_splits=NUM_FOLDS)

    gradient_boosting_model = GradientBoostingRegressor(random_state=1, n_estimators=20)
    random_forest_model = RandomForestRegressor(random_state=1, n_estimators=20)
    linear_regression_model = linear_model.LinearRegression()
    voting_model = VotingRegressor(estimators=[('gb', gradient_boosting_model), ('rf', random_forest_model), ('lr', linear_regression_model)])
    mse_scorer = make_scorer(mean_squared_error)

    results = cross_val_score(voting_model, X, y.values.ravel(), scoring=mse_scorer, cv=kfold)
    print(f"Boosting - MSE Array: {results}")

    final_results = f"Voting - Mean MSE over {NUM_FOLDS} folds: {np.mean(results)}"
    print(final_results)
    return(final_results)

starting_csv = "steam.csv"
steam_data_cleaner(starting_csv)
clean_csv = "steam_cleaned.csv"
df = steam_file_processor(clean_csv)
NUM_FOLDS = 5

#Running and timing Regression
regression_start = datetime.now()
regression_results = steam_learning_regression(df, NUM_FOLDS)
regression_end = datetime.now()
regression_total_time = regression_end - regression_start
print('Regression Total Time: ', regression_total_time)


#Running and timing Decision Tree
tree_start = datetime.now()
tree_results = steam_learning_tree(df, NUM_FOLDS)
tree_end = datetime.now()
tree_total_time = tree_end - tree_start
print('Decision Tree Total Time: ', tree_total_time)

#Running and timing Random Forest
forest_start = datetime.now()
forest_results = steam_learning_forest(df, NUM_FOLDS)
forest_end = datetime.now()
forest_total_time = forest_end - forest_start
print('Random Forest Total Time: ', forest_total_time)

#Running and timing Bagging
bagging_start = datetime.now()
bagging_results = steam_learning_bagging(df, NUM_FOLDS)
bagging_end = datetime.now()
bagging_total_time = bagging_end - bagging_start
print('Bagging Total Time: ', bagging_total_time)

#Running and timing Boosting
boosting_start = datetime.now()
boosting_results = steam_learning_boosting(df, NUM_FOLDS)
boosting_end = datetime.now()
boosting_total_time = boosting_end - boosting_start
print('Boosting Total Time: ', boosting_total_time)

#Running and timing Voting
voting_start = datetime.now()
voting_results = steam_learning_voting(df, NUM_FOLDS)
voting_end = datetime.now()
voting_total_time = voting_end - voting_start
print('Voting Total Time', voting_total_time)

#Printing results again and showing scatter plots.
print("---Linear Regression---")
print(regression_results)
print('Total Time: ', regression_total_time)
print("---Tree Regression---")
print(tree_results)
print('Total Time: ', tree_total_time)
print("---Random Forest---")
print(forest_results)
print('Total Time: ', forest_total_time)
print("---Bagging---")
print(bagging_results)
print('Total Time: ', bagging_total_time)
print("---Boosting---")
print(boosting_results)
print('Total Time: ', boosting_total_time)
print("---Voting---")
print(voting_results)
print("Total Time: ", voting_total_time)