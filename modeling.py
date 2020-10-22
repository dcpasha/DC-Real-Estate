import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  # Model Validation:
# from xgboost import XGBRegressor # $ brew install libomp

# Read the Data
file_location = "/Users/pavelpotapov/PycharmProjects/DC_Real_Estate/DC_Residential_Properties.csv"
dc_data = pd.read_csv(file_location)

# Selecting target from predictor
y = dc_data['PRICE'].copy()
X = dc_data.drop(['PRICE'], axis=1)

# Divide data into training and validation
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)
# split data into training and validation data, for both features and target.
# train_size - represent the proportion of the dataset to include in the test split.
# test_size - represent the proportion of the dataset to include in the train split.
# random_state - controls the shuffling applied to the data before applying the split.

# Let's see if there are any missing values in the dataset.
X_train_full.isnull().any()
# There are none, so we do not have to drop any.

# X_train_full.shape (47605, 32)

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Select categorical columns with a low cardinality
# Cardinality - the number of unique values in a column
cardinality_low_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

# Use only selected columns
columns = numerical_cols + cardinality_low_cols
X_train = X_train_full[columns].copy()
X_valid = X_valid_full[columns].copy()

# Let's look at summary and what columns we have left in our dataset
print(X_train.describe())
print(X_train.shape)  # (47605, 23)
print(X_train.columns)

# A list of features(things that we use to to make predictions) to choose from.
# ['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL',
#        'EYB', 'STORIES', 'SALE_NUM', 'GBA', 'KITCHENS', 'FIREPLACES',
#        'LANDAREA', 'ZIPCODE', 'LATITUDE', 'LONGITUDE', 'CENSUS_TRACT', 'AC',
#        'STRUCT_D', 'CITY', 'STATE', 'WARD']
#

# We select the following features to predict our Target
# features = ['BATHRM', 'ROOMS', 'GBA', 'LANDAREA', 'FIREPLACES', 'AYB']    # works. have MAE
features = ['BATHRM', 'ROOMS', 'GBA', 'LANDAREA', 'FIREPLACES', 'AYB', 'WARD']

X_train = X_train_full[features].copy()
X_valid = X_valid_full[features].copy()

# Let's look at our categorical columns with a low cardinality
print(cardinality_low_cols)
# ['AC', 'STRUCT_D', 'CITY', 'STATE', 'WARD']
# It does not make sense to you City or State column because we are looking at properties within Washington, DC.
# We will use Ward to indicate the geographical location of each house within the city.
# Ward - administrative division of a city or borough that typically elects and is
# represented by a councilor or councilors.

# One-Hot Encoding
# We will apply One-Hot Encoding to the Ward column because it has a categorical data.
# Apply one-hot encoder to each column with categorical data
columns_to_one_hot_encode = ['WARD']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[columns_to_one_hot_encode]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[columns_to_one_hot_encode]))

# # One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# # Remove categorical columns (will replace with one-hot encoding)
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# # Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# MODEL 1:
# A Decision Tree Regression Model:
model_decision_tree = DecisionTreeRegressor(random_state=0)
model_decision_tree.fit(OH_X_train, y_train)
preds = model_decision_tree.predict(OH_X_valid)
print(mean_absolute_error(y_valid, preds))
# MAE is 309112.17062296043

# Experimenting with different models
# Underfitting and Overfitting
# The most important option is to determine the tree's depth,
# max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting.
# Let's compare MAE scores from different values for max_leaf_node
def get_mae(max_leaf_nodes, OH_X_train, OH_X_valid, y_train, y_valid):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(OH_X_train, y_train)
    pred_val = model.predict(OH_X_valid)
    mae = mean_absolute_error(y_valid, pred_val)
    return mae


# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 25, 50, 75, 100, 250, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, OH_X_train, OH_X_valid, y_train, y_valid)
    print("Max leaf nodes is %d  \t\t , and Mean Absolute Error is  %d" % (max_leaf_nodes, my_mae))
# It looks like 500 is the best value for max_leaf_nodes.
# MAE is 243,105


# Model 2:
# A Random Forest Regression Model (AKA "ensemble method")
# Ensemble methods combine the predictions of several models.
model_random_forest = RandomForestRegressor(n_estimators=100, random_state=0)
model_random_forest.fit(OH_X_train, y_train)
preds = model_random_forest.predict(OH_X_valid)
print(mean_absolute_error(y_valid, preds))
# MAE is 234565.21492900274.

# How to find n_estimators?
# n_estimators - # of trees in the forest.
# Let's define a few Random Forest Models
model_random_forest_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_random_forest_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_random_forest_3 = RandomForestRegressor(n_estimators=50, criterion='mae', random_state=0)
model_random_forest_4 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_random_forest_5 = RandomForestRegressor(n_estimators=100, min_samples_split=20, random_state=0)
model_random_forest_6 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_random_forest_7 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
model_random_forest_8 = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=0)

models = [model_random_forest_1, model_random_forest_2
          ,model_random_forest_3, model_random_forest_4,
          model_random_forest_5, model_random_forest_6, model_random_forest_7, model_random_forest_8]


# In order to find the best model define a function to find the best MAE
def model_validation(model, X_t = OH_X_train, X_v=OH_X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


for i in range(0, len(models)):
    mae = model_validation(models[i])
    print("Random Forest Model %d had MAE: %d " % (i+3, mae))

# Random Forest Model 1 had MAE: 235805
# Random Forest Model 2 had MAE: 234565
model_random_forest_3.fit(OH_X_train, y_train)
preds = model_random_forest_3.predict(OH_X_valid)
print(mean_absolute_error(y_valid, preds))


# MODEL 3:
# XGBoost (Extreme gradient boosting)
# model_xgboost = XGBRegressor()
# model_xgboost.fit(OH_X_train, y_train)
# preds = model_xgboost.predict(OH_X_valid)
# print(mean_absolute_error(y_valid, preds))


# In order to avoid Data Leakage:
# We need to distinguish training data from validation data. Validation data is meant to measure how the model performs
# on the data that it has not seen before. We can create train-test contamination by preprocessing our data before
# calling train_test_split(). If we do so, our model may get good validation scores, but performs poorly when you deploy
# it to make decisions.
# We need to exclude the validation data, from any type of fitting, including the fitting of preprocessing steps.

# We avoid train-test contamination by doing preprocessing only on the training data. Location is believed to be one
# of the most important factors in determining house prices. We can use either Zipcode or Ward to create a location
# based category for our data.

