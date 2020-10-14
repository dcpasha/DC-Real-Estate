import pandas as pd

# Storing data
file_path_residential = "/Users/pavelpotapov/Data/Real_Estate_DC/Computer_Assisted_Mass_Appraisal_-_Residential.csv"
file_path_address_points = "/Users/pavelpotapov/Data/Real_Estate_DC/Address_Points.csv"

# Loading Data
data_residential = pd.read_csv(file_path_residential,index_col='SSL')
print("Residential is done")

data_address = pd.read_csv(file_path_address_points)
# It complains about. dtype={'LOT': float}
print("Address is done")

print("Data is loaded")

# Let's look at first five rows of each data frame
data_residential.head()
data_address.head()

data_residential.shape # (108499, 38) Tells us the size (rows, columns) of the data frame
data_address.shape # (148172, 60)

# Investigate data_residential
data_residential.info()
data_residential.describe()
