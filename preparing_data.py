import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Storing data.
file_path_residential = "/Users/pavelpotapov/Data/Real_Estate_DC/Computer_Assisted_Mass_Appraisal_-_Residential.csv"
file_path_address_points = "/Users/pavelpotapov/Data/Real_Estate_DC/Address_Points.csv"

# Loading Data
data_residential = pd.read_csv(file_path_residential, index_col='SSL')
print("Residential data is loaded")
data_address = pd.read_csv(file_path_address_points, index_col='SSL', low_memory=False)
print("Address data is loaded")

# Let's look at the first five rows and shape of the Residential data frame:
data_residential.head()

# STEP 1: Investigate Residential Data
# Column description:
# SSL - The square, suffix, and lot
# AYB - Actual Year Build. USE THIS.
# EYB - Effective Date Build.
# YR_RMDL - Year Remodeled
# GBA - Gross Building Area


data_residential.info()  # Non-null count and dtypes
data_residential.shape  # Tells us the size (rows, columns) = (108499, 38) of the data frame
data_residential.describe()  # count, mean, std, and etc.

# Investigate the dataset:
# There are 38 columns. Let's investigate and exclude those that are redundant.
# Columns 'ROOF_D', 'INTWALL_D', and 'EXTWALL_D' tell us from what materials the walls and roof are made.
data_residential['INTWALL_D'].value_counts()
data_residential['EXTWALL_D'].value_counts()
data_residential['ROOF_D'].value_counts()

# Columns 'HEAT', 'EXTWALL', 'ROOF',  'INTWALL' are interger values of their counterpart columns and can be dropped.
# Let's make a list of columns that do not carry crucial infromation:
columns_to_drop = ['OBJECTID', 'HEAT', 'STRUCT', 'GRADE', 'GRADE_D', 'CNDTN', 'CNDTN_D', 'EXTWALL', 'ROOF', 'QUALIFIED',
                   'STYLE', 'STYLE_D', 'INTWALL', 'USECODE', 'GIS_LAST_MOD_DTTM', 'BLDG_NUM']

# columns_to_keep = ['BATHRM', 'HF_BATHRM', 'AC', 'HEAT_D', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL' 'EYB',
#                    'STORIES' 'SALEDATE', 'PRICE', 'GBA', 'STRUCT_D', 'KITCHENS', 'FIREPLACES', 'ROOF_D', 'INTWALL_D',
#                    'EXTWALL_D',
#                    'NUM_UNITS']

# Drop the columns that are not useful:
data_residential.drop(columns_to_drop, axis=1, inplace=True)

# Let's see if there are any empty entries in our data:
data_residential.shape # (108499, 22)
data_residential.isnull().sum()

# There are a few empty entries, but 'PRICE' and 'ROOM' columns have many missing entires.
# We cannot analyze houses that do not have a sales price and at least one room.
# It is not considered a house if it doesn't have a room. Drop entries without price and at least one room:
data_residential.dropna(subset=['PRICE', 'ROOMS'], inplace=True)
# Let's also drop entries where 'PRICE' is zero:
data_residential = data_residential.loc[(data_residential['PRICE'] != 0), :]

data_residential.astype(bool).sum(axis=0)
data_residential.isnull().sum()

# Now, it looks better. There are still a few missing entries in columns 'KITCHENS', 'FIREPLACES' and 'BEDRM'.
# It is possible to have a house without a kitchen, it is called an efficiency studio.
# If they are missing, let's assume that they do not exist. Thus, we replace missing values with 0.
data_residential.loc[:, 'FIREPLACES'].fillna(value=0, inplace=True)
data_residential.loc[:, 'KITCHENS'].fillna(value=0, inplace=True)
data_residential.loc[:, 'BEDRM'].fillna(value=0, inplace=True)

# Let's investigate 'STORIES'
data_residential['STORIES'].value_counts()
# 'STORIES' tells us how many floors a house has. If a unit is underground, it has 0 floors.
data_residential['STORIES'].value_counts()
# Let's replace missing values with 1:
data_residential.loc[:, 'STORIES'].fillna(value=1, inplace=True)

# There are some entries with missing values and a few houses with partial floors.
# For example, one and a half story home is a one story home with a partial second floor added to allow for more space.
# We will keep the partial floors in our dataset.
# If you want to round up a 1/4 floor and round down 3/4 to a partial floor, you can uncomment the code below and run it:
# def round_of_rating(number):
#     # Round a number to the closest half integer.
#    return round(number * 2) / 2
#
#
# # data_residential['STORIES'] = data_residential['STORIES'].apply(round_of_rating)


data_residential.isnull().sum()
# Let's look at the AYB (Actual Year Build), EYB (Effective Date Build), and YR_RMDL (Year Remodeled)
# There are a few missing AYB. Let's find the average difference between AYB and EYB
# AYB - Actual Year Build. USE THIS.
# EYB - Effective Date Build.
# YR_RMDL - Year Remodeled
eyb_ayb_difference = data_residential['EYB'] - data_residential['AYB']
eyb_ayb_mean_difference = eyb_ayb_difference.mean()
# The mean difference calculated is 40 years.
# If AYB is missing or zero, we will replace it with its EYB - 40.
data_residential.loc[data_residential['AYB'].isnull(), ['AYB', 'EYB']] # Shows us what entries having a missing 'AYB'
data_residential['AYB'].mask(data_residential['AYB'] == 0, inplace=True)  # convert zero to na
data_residential.loc[:, 'AYB'].fillna(data_residential['EYB'] - eyb_ayb_mean_difference, inplace=True)
data_residential['AYB'].isnull().sum()


# If YR_RMDL is missing, we will replace it with EYB because it is a newer date compared to AYB.
data_residential['YR_RMDL'].fillna(data_residential['EYB'], inplace=True)
data_residential.isnull().sum() # We do not have any missing values in our dataframe.


data_residential.info()
# Convert data types to integers where appropriate:
columns_to_int = ['BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'EYB', 'YR_RMDL', 'PRICE', 'KITCHENS',
                  'FIREPLACES']
for i in columns_to_int:
    data_residential[i] = data_residential[i].astype('int64')

# Let's look at the 'SALEDATE' column:
data_residential['SALEDATE'] = pd.to_datetime(data_residential['SALEDATE'], dayfirst=False)
data_residential["SALEDATE"].sort_values(ascending=True).head(10)
data_residential["SALEDATE"].sort_values(ascending=False).head(10)
# It looks good. We do not need to make any changes.

# The SSL - the square, suffix, and lot.
# Are there any SSL that are duplicates? Let's see if we have any duplicates of SSL.
data_residential.index.duplicated().any()
# Is there a duplicated transaction (the same 'SALEDATE') or the property was sold more than once?
print(data_residential[data_residential.index.duplicated()])
print(data_residential.loc[data_residential.index == '5890    0134', :])
# Yes, there are some duplicates transactions.
# Let's drop duplicate 'SSL' and only keep the last one.
data_residential = data_residential[~data_residential.index.duplicated(keep='last')]

# Let's make a correlation and see if we can detect any anomalies and represent them visually.
plt.style.use('seaborn-colorblind')
# Basic correlogram
sns.pairplot(data_residential[['ROOMS', 'BATHRM', 'HF_BATHRM', 'BEDRM']], kind="scatter", diag_kind='kde',
             plot_kws={'alpha': 0.5, 's': 90}, size=6)
plt.show()

# The graph shows us a few outliers:
# The ROOMS X BTHRM plot shows that there is at least one property with more than 100 rooms and less than five bathrooms.
# The BEDRM X ROOMS plot shows that there are properties that have more bedrooms than rooms.
# The BEDRM X BATHRM plot i shows that there is at least one property with more than 24 bedrooms and 24 bathrooms.
# Let's exclude these outliers from our dataset:
data_residential["FIREPLACES"].value_counts()
# There are a few houses with more than 10 fireplaces, I would also exclude them.

data_residential = data_residential[(
        (data_residential['ROOMS'] < 100) & (data_residential['ROOMS'] >= data_residential['BEDRM']) & (
        data_residential['BATHRM'] < 24) & (data_residential['FIREPLACES'] < 10))]

data_residential.loc[:, ['BATHRM', 'ROOMS', 'BEDRM', 'PRICE', 'SALEDATE','FIREPLACES']].describe()
data_residential.loc[:, ['BATHRM', 'ROOMS', 'BEDRM', 'PRICE', 'SALEDATE']].head()
data_residential.sort_values('SALEDATE', inplace=True)

data_residential.to_csv("data_residential.csv", header=True)  # To save the residential data

# STEP 2: Investigate the Address data.
# This dataset complements the residetial data with some spacial information.
# Square Suffix Lot (SSL) will be used to merger residential data and address data.
data_address.info()
data_address.head()

# Let's check if there are any duplicated indexes ('SSL') and treat them in the same way as with the residential information.
data_address.index.duplicated().any()  # True, so let's drop the duplicates:
data_address = data_address[~data_address.index.duplicated(keep='last')]

# Choosing which columns to keep:
address_columns = ["FULLADDRESS", "CITY", "STATE", "ZIPCODE", "LATITUDE", "LONGITUDE",
                   "ASSESSMENT_NBHD", "CENSUS_TRACT", 'CENSUS_BLOCK', "WARD"]

data_address = data_address[address_columns]


dataset = pd.merge(data_residential, data_address, how="left", on="SSL")
# Drop entries with missing FULLADDRESS:
dataset.dropna(subset=['FULLADDRESS'], inplace=True)
# Saving our cleaned dataset to a csv file:
dataset.to_csv("DC_Residential_Properties.csv", header=True)

################################
# Useful Commands for analysis:
# Comparing Columns:
# data_residential.loc[:, ['AYB', 'EYB', 'YR_RMDL']].head(100)

# Creating a new column based on a boolean condition two other columns:
# cond = (aye_eyb['AYB'] > aye_eyb['EYB'])
# aye_eyb['AYB vs EYB'] = np.where(cond, "True", "False")
# aye_eyb.loc[aye_eyb['AYB vs EYB'] == 'True',]
# aye_eyb.loc[aye_eyb['AYB'],].value_counts()

# data_address.loc['SSL'=='0202    0086']
# Looking at a certain address at a specified SSL value:
# data_address.loc[data_address['SSL']=='0070    2412 ', ['FULLADDRESS']]

# data_residential.loc[['2113    0136']] # shows a row of a dataframe at the specified index.
# data_residential.loc[['2113    0136'],['AYB','EYB']]
