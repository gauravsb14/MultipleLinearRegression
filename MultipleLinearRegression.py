import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# import the data
df = pd.read_csv("/home/gaurav/AI/Proiba_ML/Regression/RegressionData/50_Startups.csv")
print(df.head())


# label encoding for categorical data
#labels the categorical variables with dummy variables as 0,1 or 2.
la = LabelEncoder()
df["State"] = la.fit_transform(df["State"])

# ?checking shape and df after label encoding
print(df.head())
print(df.shape)

# Defining training and testing dataset
X = df.iloc[:,0:4]
Y = df["Profit"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .1, random_state = 1)

# creating model
lr = LinearRegression()


# training model
lr.fit(X_train,Y_train)

# predicting the value
pred = lr.predict(X_test)
print("Prediction value: {}".format(pred))

# checking score
print("Prediction value: {}".format(lr.score(X_test, Y_test)))