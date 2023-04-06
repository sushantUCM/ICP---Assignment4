from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('train_preprocessed.csv')
test_data = pd.read_csv('test_preprocessed.csv')

# Find the correlation between Survived and Sex
corr = train_data['Survived'].corr(train_data['Sex'].astype('category').cat.codes)
print("Correlation between Survived and Sex: ",corr)


print('a. Do you think we should keep this feature?')
print('Yes, we should keep this feature as it has a correlation of', corr, 'with the target variable but some other features can be dropped as they have very less correlation with the target variable.')

# Do at least two visualizations to describe the data
# Histogram of age
train_data['Age'].plot.hist(title='Histogram of Age')
plt.show()

# Scatter plot of age and fare
train_data.plot.scatter(x='Age', y='Fare', title='Scatter plot of Age and Fare')
plt.show()

# Plot between age and survived
train_data.plot.scatter(x='Age', y='Survived', title='Scatter plot of Age and Survived')
plt.show()

# Implement Naïve Bayes method using scikit-learn library and report the accuracy
# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets which is already preprocessed
# drop PassengerId column as it is not required in test data
test_data.drop('PassengerId', axis=1, inplace=True)
# drop Survived column in train data as it is the target variable
train_data.drop('Survived', axis=1, inplace=True)

# Fit the model with the training data
model.fit(train_data.drop('Embarked', axis=1), train_data['Embarked'])
# Predict the response for test dataset
y_pred = model.predict(test_data.drop('Embarked', axis=1))
# Calculate the accuracy of the model
print("Accuracy of the model: ", accuracy_score(test_data['Embarked'], y_pred))
