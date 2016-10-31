# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:29:45 2016

"""

import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

train = pandas.read_csv("C:\\Users\\bingl\\Dropbox\\job\\divvy_prediction\\train.txt",sep='\t', dtype={'id': np.int32})
test = pandas.read_csv("C:\\Users\\bingl\\Dropbox\\job\\divvy_prediction\\test.txt", sep='\t', dtype={'id': np.int32})
print train.head(3)
print train.describe()

# 116 columns of categorical features, plus 14 continuous features plus log loss
# First try with partial data to speed up the learning process
train = train.drop(['daylabel', 'year', 'month', 'day'], axis=1)
train["count"].apply(lambda x: 1 if x==0 else x)
train["count"] = numpy.log1p(train["count"])

split = 5
n = 5
cols = train.columns

#for i in range(n):
#    fg,ax = plt.subplots(nrows=1,ncols=1,figsize=(12, 8))
#    sns.violinplot(y = cols[i+5], data = train)
#    plt.savefig("C:\\Users\\bingl\\Dropbox\\job\\divvy_prediction\\violin_plot_"+cols[i+5]+".png")

train_cont = train.iloc[:, split:]
train_cat = train.iloc[:, :split]
cats = train_cat.columns
conts = train_cont.columns

# now labels contain all possible categories for each feature from 1 to 116
labels = {}
for i, col in enumerate(cats):
    label_in_train = train[col].unique()
    label_in_test = test[col].unique()
    labels[col] = list(set(label_in_train) | set(label_in_test))

#One hot encode all categorical attributes
train_cat_transformed = []

for i, col in enumerate(cats):
    print i, col, len(labels[col])
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[col])
    
    feature = label_encoder.transform(train_cat.iloc[:,i]).reshape(train_cat.shape[0], 1)
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[col]))
    feature = onehot_encoder.fit_transform(feature)
    train_cat_transformed.append(feature)
    
encoded_cats = numpy.column_stack(train_cat_transformed)
train_encoded = numpy.concatenate((encoded_cats,train_cont.values),axis=1)

#scale the X feature for both training set and testing set
X_train = train_encoded[:, :-1]
Y_train = train_encoded[:, -1]

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

print "after the one-hot-encoding, the number of all features is: \n"
print encoded_cats.shape[1]
del train
del train_cont
del encoded_cats

#get the number of rows and columns
r, c = train_encoded.shape
#model = LinearRegression(n_jobs=-1)
#model.fit(X_train, Y_train)
model = svm.SVR()
model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
result = np.mean((Y_train - Y_train_pred)**2)

print("average square error (training set) %s" % result)
print("mean_absolute_error %s" % mean_absolute_error(Y_train, Y_train_pred))

plt.figure(6)
plt.scatter(Y_train_pred, Y_train, c = 'b')
x = np.linspace(0., 8., 100)
y = x
plt.xlabel("predictions in log scale")
plt.ylabel("log(counts)")
plt.plot(x, y, c= 'red', linewidth = 2.5)
plt.savefig("C:\\Users\\bingl\\Dropbox\\job\\divvy_prediction\\training_result_logscale_svm.png")


plt.figure(7)
plt.scatter(np.exp(Y_train_pred), np.exp(Y_train), c = 'b')
x = np.linspace(-100., 1000., 200)
y = x
plt.xlabel("predictions")
plt.ylabel("original counts")
plt.plot(x, y, c= 'red', linewidth = 2.5)
plt.savefig("C:\\Users\\bingl\\Dropbox\\job\\divvy_prediction\\training_result_svm.png")

plt.show()
