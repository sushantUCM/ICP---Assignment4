import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# a. read glass.csv file as a dataframe
glass_data = pd.read_csv('glass.csv')

# b. Use train_test_split to create training and testing part
# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(glass_data.drop('Type', axis=1), glass_data['Type'], test_size=0.3, random_state=42)

# implement Naïve Bayes method using scikit-learn library and Evaluate the model on testing part using score and classification_report
# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
print("Accuracy of the Naive Bayes model: ", accuracy_score(y_test, y_pred))

# Print classification report
print('Classification Report for Naive Bayes model: ')
print(classification_report(y_test, y_pred))


# Use SVM method using scikit-learn library and Evaluate the model on testing part using score and classification_report
# Create a SVM Classifier
model = SVC(kernel='linear')

# Train the model using the training sets
model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
print("Accuracy of the SVM model: ", accuracy_score(y_test, y_pred))

# Print classification report
print('Classification Report for SVM model: ')
print(classification_report(y_test, y_pred))


# Do at least two visualizations to describe or show correlations in the Glass Dataset
# Histogram of refractive index
glass_data['RI'].plot.hist(title='Histogram of Refractive Index')
plt.show()

# Scatter plot of refractive index and Ca
glass_data.plot.scatter(x='RI', y='Ca', title='Scatter plot of Refractive Index and Ca')
plt.show()

print('Which algorithm you got better accuracy? Can you justify why?')
# accuracy of Naive Bayes model:  0.3076923076923077
# accuracy of SVM model:  0.676923076923077
print('SVM model has better accuracy than Naive Bayes model. This is because SVM model tries to find the best possible decision boundary between the data points of different classes. It tries to maximize the margin between the decision boundary and the data points. On the other hand, Naive Bayes model assumes that the features are independent of each other and tries to find the probability of the data point belonging to a particular class. Hence, SVM model has better accuracy than Naive Bayes model.')