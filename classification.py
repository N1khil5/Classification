import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Setting a random seed for the file to get the same results with every run.
np.random.seed(0)

data = pd.read_csv('adult.csv')
'''
Dropping column
References: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
'''
newData = data.drop(labels=None, axis=1, columns='fnlwgt')

# Q1.1(i) Printing number of instances
print("number of instances = ", len(newData))

''' 
Q1.1 (ii) Generating number of missing values
References: 
1. https://medium.com/analytics-vidhya/python-finding-missing-values-in-a-data-frame-3030aaf0e4fd
2. https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
'''
missingvalues = newData.isnull().sum().sum()
print("missingvalues =", missingvalues)

# Finding the size of all the attribute values
allValues = newData.size
print("all attribute values = ", allValues)

# Calculating attributes without null attributes
withoutNa = len(newData.dropna())
print("without Null values = ", withoutNa)

# Q1.1 (iv) Number of instances with missing values
totalWithoutNA = len(newData) - withoutNa
print("Attributes without NA =", totalWithoutNA)

# Q1.1 (iii) and Q1.1 (v) require fractions and both of those are represented in the report.

# Q1.2 Printing Data before the LabelEncoder
for i in newData.columns:
    print("Column = ", i)
    print(pd.unique(newData.loc[:, i]))

# Label encoder for the data
labelencoder = LabelEncoder()
newDataEncoded = newData.apply(LabelEncoder().fit_transform)

# Printing set of all possible discrete values for each attribute
for i in newDataEncoded.columns:
    print("Column = ", i)
    print(pd.unique(newDataEncoded.loc[:, i]))

# Q1.3 Removing any missing values for the decision tree classifier.
decisionTreeData = newDataEncoded.dropna()

# Decision Tree created from the sample given in Practical 3
X = decisionTreeData.drop(['class'], axis=1)
y = decisionTreeData['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
decisionTreeModel = DecisionTreeClassifier(random_state=1)
decisionTreeModel.fit(X_train, y_train)
decisionTreePredict = decisionTreeModel.predict(X_test)

# Finding the accuracy of the decision tree
accuracy = accuracy_score(y_test, decisionTreePredict)

# Computing the error of the decision tree
error = 1 - accuracy
print("Error =", error)

# Q1.4 (i) Finding rows of data with missing values to create D'
findMissing = newData.isnull()
rowsWithMissing = findMissing.any(axis=1)
Dprime = newData[rowsWithMissing]
print(Dprime)

# Q1.4 (ii) Dropping NA values and creating a dataset of non NA values in the same size as D'.
DwithoutNa = newData.dropna()
D = DwithoutNa.sample(len(Dprime))
print(D)

# Combining both D and D' to form one dataset.
combinedDDprime = Dprime.append(D)
print("D+D'=", combinedDDprime)

# Creating D'1 where NA values are replaced with "missing"
dprime1 = combinedDDprime.sample(3620)
dprime1 = dprime1.fillna("missing")
print("D'1= ", dprime1)

# Creating D'2 where NA values are replaced with most popular value for each attribute.
dprime2 = combinedDDprime.drop(dprime1.index)
for i in dprime2.columns:
    modeForColumn = dprime2[i].mode()[0]
    dprime2[i].fillna(modeForColumn, inplace=True)
print(dprime2)

# Creating a test set from the original dataset D without the data used for training.
SmallSampleFromD = newData.drop(combinedDDprime.index)
SmallSampleFromD = SmallSampleFromD.apply(LabelEncoder().fit_transform)
SmallSampleFromD = SmallSampleFromD.sample(1086)
SmallSampleX = SmallSampleFromD.drop(['class'], axis=1)
SmallSampleY = SmallSampleFromD['class']

# Implementing a decision tree for D'1 and comparing to test set created above
dprime1Enc = dprime1.apply(LabelEncoder().fit_transform)
dprime1EncX = dprime1Enc.drop(['class'], axis=1)
dprime1EncY = dprime1Enc['class']
decisionTreeModelDP1 = DecisionTreeClassifier(random_state=1)
decisionTreeDprime1 = decisionTreeModelDP1.fit(dprime1EncX, dprime1EncY)
decisionTreePredictDprime1 = decisionTreeDprime1.predict(SmallSampleX)

# Outputting the error of the D'1 decision tree.
accuracyDprime1 = accuracy_score(SmallSampleY, decisionTreePredictDprime1)
errorDP1 = 1 - accuracyDprime1
print("ERROR D'1 =", errorDP1)

# Implementing a decision tree for D'2 and comparing to test set created above
dprime2Enc = dprime2.apply(LabelEncoder().fit_transform)
dprime2EncX = dprime2Enc.drop(['class'], axis=1)
dprime2EncY = dprime2Enc['class']
decisionTreeModelDP2 = DecisionTreeClassifier(random_state=1)
decisionTreeDprime2 = decisionTreeModelDP2.fit(dprime2EncX, dprime2EncY)
decisionTreePredictDprime2 = decisionTreeDprime2.predict(SmallSampleX)

# Outputting the error for the D'2 decision tree.
accuracyDprime2 = accuracy_score(SmallSampleY, decisionTreePredictDprime2)
errorDP2 = 1 - accuracyDprime2
print("ERROR D'2 =", errorDP2)
