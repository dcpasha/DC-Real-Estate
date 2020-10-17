import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Storing data
file_path_residential = "/Users/pavelpotapov/Data/Real_Estate_DC/Computer_Assisted_Mass_Appraisal_-_Residential.csv"
file_path_address_points = "/Users/pavelpotapov/Data/Real_Estate_DC/Address_Points.csv"

# Loading Data
data_residential = pd.read_csv(file_path_residential, index_col='SSL')
print("Residential data is loaded")
data_address = pd.read_csv(file_path_address_points, index_col='SSL')
# It complains about. dtype={'LOT': float}
print("Address data is loaded")

# Let's look at first five rows and shape of the Residential data frame
# data_residential.head()

# STEP 1:Investigate Residential Data
# data_residential.info()  # Non-null count and dtypes
# data_residential.shape (108499, 38) Tells us the size (rows, columns) of the data frame
# data_residential.describe()  # mean, std,

# Investigate the dataset:
# There are 38 columns. Let's investigate them and exclude those that are redundant.
# The columns 'ROOF_D', 'INTWALL_D', and 'EXTWALL_D' tell us from what materials the walls and roof are made.
# data_residential['INTWALL_D'].value_counts()
# data_residential['EXTWALL_D'].value_counts()
# data_residential['ROOF_D'].value_counts()

# The following columns: 'HEAT', 'EXTWALL', 'ROOF',  'INTWALL' are interger values of their counterpart columns and can be dropped.
# data_residential['ROOF_D'].value_counts()
# Let's make a list of columns that does not carry a lot information and drop them.
columns_to_drop = ['OBJECTID', 'HEAT', 'STRUCT', 'GRADE', 'GRADE_D', 'CNDTN', 'CNDTN_D', 'EXTWALL', 'ROOF', 'QUALIFIED',
                   'STYLE', 'STYLE_D', 'INTWALL', 'USECODE', 'GIS_LAST_MOD_DTTM']

columns_to_keep = ['BATHRM', 'HF_BATHRM', 'AC', 'HEAT_D', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL' 'EYB',
                   'STORIES' 'SALEDATE', 'PRICE',
                   'GBA', 'STRUCT_D', 'KITCHENS', 'FIREPLACES', 'ROOF_D', 'INTWALL_D', 'EXTWALL_D',
                   'NUM_UNITS']  # I don't like QUALIFIED.

# for i in range(len(columns_to_drop)):
#     if(columns_to_drop[i] in columns_to_keep):
#         print(columns_to_drop[i])
# #
# # Dropping the columns that are not useful
data_residential.drop(columns_to_drop, axis=1, inplace=True)

# Let's see if there are any empty entries in our data.
# data_residential.shape # (108499, 23)
# data_residential.isnull().sum()

# There are a few empty entries, but 'PRICE' and 'ROOM' columns miss a lot of entires.
# We cannot analysis houses that do not have a sales price, and at least one room.
# It does not constitues it as a house.
# Drop entries without price and room.
data_residential.dropna(subset=['PRICE', 'ROOMS'], inplace=True)
data_residential.isnull().sum()

# Now it looks better. There are still a few missing entries in columns 'KITCHENS', 'FIREPLACES' and 'BEDRM'.
# If they are missing, let's assume that they do not exist. Thus, we replace missing values with 0.
data_residential['FIREPLACES'] = data_residential['FIREPLACES'].fillna(value=0)
data_residential['KITCHENS'] = data_residential['KITCHENS'].fillna(value=0)
data_residential['BEDRM'] = data_residential['BEDRM'].fillna(value=0)

# # Let's investigate 'STORIES'
# data_residential['STORIES'].value_counts()
# # 'STORIES' tells us how many floors a house has.
# # There are some entries with missing values and a few houses with partial floors.
# # For example, one and a half story home is a one story home with a partial second floor added to allow for more space.
# # We will keep the partial floors in our dataset, but we will round up a 1/4 floor and round down 3/4 to a partial floor.

# Let's replace missing values with 1 and give one floor to anybody who has less than 1 floor in the dataset.
data_residential['STORIES'] = data_residential['STORIES'].fillna(value=1)
data_residential.loc[data_residential['STORIES'] < 1, :] = 1


def round_of_rating(number):
    # Round a number to the closest half integer.
    return round(number * 2) / 2


data_residential['STORIES'] = data_residential['STORIES'].apply(round_of_rating)
data_residential['STORIES'].value_counts()

data_residential.isnull().sum()
data_residential.info()
# Let's look at the AYB (Actual Year Build), EYB (Effective Date Build), and YR_RMDL (Year Remodeled)
# AYB - Actual Year Build. USE THIS.
# EYB - Effective Date Build.
# YR_RMDL - Year Remodeled
# If AYB is missing, we will replace it with its EYB.
data_residential.loc[data_residential['AYB'].isnull(), ['AYB', 'EYB']]
data_residential['AYB'].fillna(data_residential['EYB'], inplace=True)

# # data_residential.loc[['2113    0136']] # shows a row of a df at the specified index
# # data_residential.loc[['2113    0136'],['AYB','EYB']]

# If YR_RMDL is missing, we will replace it with EYB because it is a newer date compared to AYB.
data_residential['YR_RMDL'].fillna(data_residential['EYB'], inplace=True)
# data_residential.isnull().sum()

# Convert data types to int where appropriate; DOES NOT WORK
columns_to_int = ['PRICE', 'BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'EYB', 'YR_RMDL', 'KITCHENS',
                  'FIREPLACES']
for i in columns_to_int:
    data_residential[i] = data_residential[i].astype('int64')

# Let's look at the sale date.
data_residential['SALEDATE'] = pd.to_datetime(data_residential['SALEDATE'], dayfirst=False)
# data_residential["SALEDATE"].sort_values(ascending=True).head(5)
# data_residential["SALEDATE"].sort_values(ascending=False).head(5)
# It looks good. We do not need to make any changes.


# The SSL - the square, suffix, and lot.
# Are there any SSL that are duplicates? Let's see if we have any duplicates of SSL.
data_residential['SSL'].duplicated().any()  # Yes, we have some duplicates
# Is a duplicated transaction(the same 'SALES' date) or the property was sold more than once?
# Let's investigate further.
print(data_residential[data_residential.index.duplicated()])
print(data_residential.loc[data_residential.index == '5890    0134', :])
# Yes, there are some duplicates transactions, so we assume that all of them are.
# Let's drop duplicate 'SSL' and only keep the last one.
data_residential = data_residential[~data_residential.index.duplicated(keep='last')]

# # Let's make a correlation and see if we can detect any anomalies visually.
# plt.style.use('seaborn-colorblind')
# # Basic correlogram
# sns.pairplot(data_residential[['ROOMS', 'BATHRM', 'HF_BATHRM', 'BEDRM']], kind="scatter", diag_kind='kde',
#              plot_kws={'alpha': 0.5, 's': 90}, size=6)
# plt.show()

# The graph shows us a few outliers:
# The ROOMS X BTHRM plot shows that there is at least one property with more than 100 rooms and less than five bathrooms
# The BEDRM X ROOMS plot shows that there are properties that have more bedrooms than rooms.
# The BEDRM X BATHRM plot i shows that there is at least one property with more than 24 bedrooms and 24 bathrooms.
# Let's exclude these outliers from our dataset
data_residential = data_residential[(
            (data_residential["ROOMS"] < 100) & (data_residential["ROOMS"] >= data_residential["BEDRM"]) & (
                data_residential["BATHRM"] < 24))]

# TODO: redo a graph and do a new graph to make sure it looks good.
# STEP 2: Investigate the Address data.
# This dataset complements residetial data with spacial information
# data_address.shape  # (148172, 60)
# Square Suffix Lot (SSL) will be used to merger residential data and address data.
data_address.info()
data_address.head()
# Let's check if there are any duplicated indexes ('SSL') and treat them the same way we did with the residential information.
data_address.index.duplicated().any() # True, so let's drop the duplicates
data_address = data_address[~data_address.index.duplicated(keep='last')]

# TODO: Choose what columns I want to keep
address_columns = ["FULLADDRESS","CITY","STATE","ZIPCODE","NATIONALGRID","LATITUDE","LONGITUDE","ASSESSMENT_NBHD","ASSESSMENT_SUBNBHD","CENSUS_TRACT","CENSUS_BLOCK","WARD"]
data_address=data_address[address_columns]

# test = data_residential.copy()
############################
# Comparing Columns:
# data_residential.loc[:, ['AYB', 'EYB', 'YR_RMDL']].head(100)
# aye_eyb = data_residential.loc[:, ['AYB', 'EYB', 'YR_RMDL']].head(100)
#
# cond = (aye_eyb['AYB'] > aye_eyb['EYB'])
# aye_eyb['AYB vs EYB'] = np.where(cond, "True", "False")
# aye_eyb['AYB vs EYB'].value_counts()
#
# aye_eyb.loc[aye_eyb['AYB vs EYB'] == 'True',]
# aye_eyb.loc[aye_eyb['AYB'],].value_counts()

# data_address.loc['SSL'=='0202    0086']
# data_address.loc[:,'SSL'].head()
# data_address.loc[data_address['SSL']=='0070    2412 ', ['FULLADDRESS']]


