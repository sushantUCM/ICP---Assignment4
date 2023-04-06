from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Read the provided CSV file ‘data.csv’
data = pd.read_csv('data.csv')

# Show the basic statistical description about the data
print(data.describe(), '\n')

# Check if the data has null values.
print("Null values in the data: \n", data.isnull().sum(), '\n')
# a. Replace the null values with the mean
data.fillna(data.mean(), inplace=True)
print("Null values in the data after replacing with mean: \n", data.isnull().sum(), '\n')
print(data, '\n')

# Select at least two columns and aggregate the data using: min, max, count, mean
# selecting two columns pulse and calories
print("Aggregating the data using min, max, count, mean: \n", data[['Pulse', 'Calories']].agg(['min', 'max', 'count', 'mean']), '\n')

# Filter the dataframe to select the rows with calories values between 500 and 1000
print("Filtering the dataframe to select the rows with calories values between 500 and 1000: \n", data[(data['Calories'] > 500) & (data['Calories'] < 1000)], '\n')

# Filter the dataframe to select the rows with calories values > 500 and pulse < 100
print("Filtering the dataframe to select the rows with calories values > 500 and pulse < 100: \n", data[(data['Calories'] > 500) & (data['Pulse'] < 100)], '\n')

# Create a new “df_modified” dataframe that contains all the columns from df except for “Maxpulse”
df_modified = data.drop('Maxpulse', axis=1)
print("New dataframe after dropping Maxpulse column: \n", df_modified, '\n')

# Delete the “Maxpulse” column from the main df dataframe
data.drop('Maxpulse', axis=1, inplace=True)
print("Dataframe after dropping Maxpulse column: \n", data, '\n')

# Convert the datatype of Calories column to int datatype
data['Calories'] = data['Calories'].astype(int)
print("Data types of all columns after converting Calories to int: \n", data.dtypes, '\n')

# Using pandas create a scatter plot for the two columns (Duration and Calories)
data.plot.scatter(x='Duration', y='Calories', title='Scatter plot for Duration and Calories')
plt.show()