import pandas as pd
import sklearn

file_location = "/Users/pavelpotapov/PycharmProjects/DC_Real_Estate/DC_Residential_Properties.csv"
dc_data = pd.read_csv(file_location)

# Basic Data Exploration
# shows a summary of the data in our dataframe
dc_data.describe()
# shows all columns in the dataframe
dc_data.columns
# shows how many (rows, columns) in the dataframe
dc_data.shape  # (59847, 34)

# A Simple Decision Tree:
# TODO: We need to predict for houses that are not in our dataframe => exclude them.
# dropna drops missing values
dc_data.dropna(axis=0, inplace=True)  # (59515, 34)

# Selecting the Prediction Target
y = dc_data['PRICE']

# Choosing features(things that we use to to make predictions)
all_columns = ['SSL', 'BATHRM', 'HF_BATHRM', 'HEAT_D', 'AC', 'NUM_UNITS', 'ROOMS',
               'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'SALEDATE', 'PRICE',
               'SALE_NUM', 'GBA', 'BLDG_NUM', 'STRUCT_D', 'EXTWALL_D', 'ROOF_D',
               'INTWALL_D', 'KITCHENS', 'FIREPLACES', 'LANDAREA', 'FULLADDRESS',
               'CITY', 'STATE', 'ZIPCODE', 'LATITUDE', 'LONGITUDE', 'ASSESSMENT_NBHD',
               'CENSUS_TRACT', 'CENSUS_BLOCK', 'WARD']

dc_features = ['BATHRM', 'ROOMS', 'GBA', 'LANDAREA', 'FIREPLACES', 'AYB']
X = dc_data[dc_features]

# Review the data that we will used to predic house prices
X.describe()
X.head()

dc_data.loc[:,dc_data['AYB']==0]

# Building a Decision Tree Model: BAD because we have an issue with "In=Sample" Scores
from sklearn.tree import DecisionTreeRegressor

# Define model. You should specify a number for random_state
# to make sure that you have the same results each run
dc_model = DecisionTreeRegressor(random_state=1)

# Fit model:
dc_model.fit(X, y)

# TODO: We need to predict for houses that are not in our dataframe => exclude them.
# Results:
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(dc_model.predict(X.head()))

# Model Validation:
from sklearn.metrics import mean_absolute_error

predicted_home_prices = dc_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
# Mean absolute error
# MAE is 14,235.381642395223 using ['BATHRM', 'ROOMS', 'GBA', 'LANDAREA', 'FIREPLACES']

# Building a Decision Tree Regressor Model:
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define model
dc_model = DecisionTreeRegressor()
# Fit model
dc_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = dc_model.predict(val_X)
print(mean_absolute_error(val_y,val_predictions))
# MAE is 354,390 using ['BATHRM', 'ROOMS', 'GBA', 'LANDAREA', 'FIREPLACES']

# Underfitting and Overfitting
# The most important option is to determine the tree's depth,
# max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting.
# Let's compare MAE scores from different values for max_leaf_node
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
       model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
       model.fit(train_X, train_y)
       pred_val = model.predict(val_X)
       mae = mean_absolute_error(val_y, pred_val)
       return (mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 25, 50, 75,100, 250, 500, 1000, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# It looks like 75 is the best value for max_leaf_nodes.
# MAE is 273,286


# Building a Random Forest Model: from scratch



# CONCERNS:
# I don't like that there is a house with 45 FIREPLACES!

# QUESTIONS:
# I think we should include: zipcode or ward.
dc_data['ZIPCODE'].nunique()  # 21
dc_data['WARD'].nunique()  # 8
# Should we treat zipcode and ward as a categorical variable?
