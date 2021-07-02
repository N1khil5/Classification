# Classification

This project is a Data Mining coursework that uses the Adult dataset from the UCI Machine Learning Repository to predict whether the income of an individual exceeds 50K per year based on 14 attributes. 

The first part of the coursework is to create a table with the number of instances, number of missing values, fraction of missing values over all attribute values, number of instances with missing values and fraction of instances with missing values over all instances. 

The second part of the coursework converts the attributes to nominal and prints the discrete values of each attribute.

The third part builds a decision tree for classifying and individual to one of the <=50k and > 50k categories and compute the error rate. 

The last part of this coursework trains two decision trees using two different approaches to handle missing values. The first way is to create a new value "missing" and the second method is by using the most popular value for all missing values of each attribute. 


(Line 11)
The source code reads data from the Adult.csv data file.

(Line 16)
preprocesses the data by removing the "fnlwgt" column. 

(Line 27-40)
The source code prints out a number of outputs for the table
as required in Q1.1. 

(Line 50-51)
Attributes were converted to nominal values as per Q1.2.

(Line 59-74)
The code then creates a decision tree without missing values as per Q1.3 with the help 
of the Scikit-learn. A test size of 30% is used here. Accuracy and error are also computed.

(Line 77-80)
Creating the modified dataset D' with all the instances with at least one missing value.

(Line 83-85)
Creating a modified dataset D of only isntances without missing values.   

(Line 88-89)
Combining D and D' from above.

(Line 92-94)
Sampling a modified subset of the data D'1 and replacing NA values with missing.

(Line 97-101)
Sampling a modified subset of the data D'2, which does not contain D'1. Here the NA values
are replacd with the mode of that column. 

(Line 104-108)
Creating a small testing sample that will be used for the decision tree for D'1 and D'2.
This testing sample comes from the original dataset and is encoded. 

(Line 111-121)
Here, the decision tree for D'1 is calculated and the error is printed to the console. 
X_train is the encoded D'1 sample without the class column, Y_train is the class column
of the encoded D'1 sample, X_test and Y_test are from the original dataset D that we created
in line 104-108.

(Line 124-134)
A decision tree is computed for D'2 and error is printed to the console. Format is similar
to the way D'1's decision tree was calculated, X_train and Y_train were from the D'2 Encoded
dataset while X_test and Y_test are from the sample data in line 104-108.
