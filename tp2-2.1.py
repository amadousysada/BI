#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from scipy.io import arff
from sklearn.impute import SimpleImputer

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

dummycl = DummyClassifier(strategy="most_frequent")
gmb = GaussianNB()
dectree = tree.DecisionTreeClassifier()
logreg = LogisticRegression(solver="liblinear")
svc = svm.SVC(gamma='scale')

<<<<<<< HEAD
data, meta = arff.loadarff("crx.arff")

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit_transform(data)

standardscaler = preprocessing.StandardScaler()

X_norm = standardscaler.fit_transform(data)

data, meta = arff.loadarff('crx.arff')
df = pd.DataFrame(data)

plt.hist(data['class'], bins='auto')
plt.savefig("distribution des classes positives et negatives")


percent=df.groupby(['class']).agg({'class':'count'})
percent = percent.divide(len(df['class']))*100



lst_classif = [dummycl, gmb, dectree, logreg, svc]
lst_classif_names = ['Dummy', 'Naive Bayes', 'Decision tree', 'Logistic regression', 'SVM']


def accuracy_score(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        scores = cross_val_score(clf, X, y, cv=5)
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def confusion_matrix(lst_classif,lst_classif_names,X,y):
    for clf,name_clf in zip(lst_classif,lst_classif_names):
        predicted = cross_val_predict(clf, X, y, cv=5) 
        print("Accuracy of "+name_clf+" classifier on cross-validation: %0.2f" % metrics.accuracy_score(y, predicted))
        print(metrics.confusion_matrix(y, predicted))

# Replace missing values by mean and scale numeric values
data_num = df.select_dtypes(include='float64')
labels = data_num.columns
#Imputation des valeurs manquantes
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

data_num = imp_mean.fit_transform(data_num)

#Standardisation
standardscaler = preprocessing.StandardScaler()

X = standardscaler.fit_transform(data_num)

y= df['class']

accuracy_score(lst_classif,lst_classif_names,X,y)

# Replace missing values by mean and discretize categorical values
data_cat = df.select_dtypes(exclude='float64').drop('class',axis=1)
imp_most_freq = SimpleImputer(missing_values='?', strategy='most_frequent')
data_cat = imp_most_freq.fit_transform(data_cat)
X = pd.get_dummies(pd.DataFrame(data_cat))
print("\n\n Categorical classement")
accuracy_score(lst_classif,lst_classif_names,X,y)

df[labels] = standardscaler.fit_transform(df[labels])
print("\n\n Toute les donnees")
d=pd.concat(df[labels],X)
print type(df[labels])
exit()
accuracy_score(lst_classif,lst_classif_names,df,y)
