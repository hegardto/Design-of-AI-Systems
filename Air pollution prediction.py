# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

beijing_df = pd.read_csv('Beijing_labeled.csv')
chengdu_df = pd.read_csv('Chengdu_labeled.csv')
guangzhou_df = pd.read_csv('Guangzhou_labeled.csv')
shanghai_df = pd.read_csv('Shanghai_labeled.csv')
shenyang_df = pd.read_csv('Shenyang_labeled.csv')
train_df = pd.read_csv('Beijing_labeled.csv')
train_df = train_df.append(pd.read_csv('Shenyang_labeled.csv'))

#Feature selection
X = train_df.iloc[:,0:9]  #independent columns
y = train_df.iloc[:,10]    #target column i.e PM_HIGH

#Calculate correlations between all features in dataset
corrmat = train_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(11,11))

#Plot heat map
g=sns.heatmap(train_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.savefig('Heatmap', dpi=400)

#Drop variables showing correlation > 0.1
train_df2 = train_df.drop(columns=['season', 'PRES', 'TEMP', 'precipitation', 'cbwd_NE'])
X = train_df2.drop(['PM_HIGH'],axis=1)
X = X.to_numpy()
y = train_df2['PM_HIGH']
y = y.to_numpy()

class KNN(object):
    def __init__(self, k):
        self.k = k
    
    #Introduce classvariables for storing training data as wel as test data
    X_train = []
    y_train = []
    dataset = []
    dataset2 = []
    x_test = []
   
    #Save the training data to the object, in dataset and dataset2
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        df = pd.DataFrame(data=X_train, columns=["1","2","3","4","5"])
        self.dataset = df.to_numpy()
        self.dataset = KNN.scale(self.dataset, 1,0)
        df['6']= y_train
        self.dataset2 = df.to_numpy()

    #Normalize the dataset to allow for calculation of euclidean distance
    def scale(data, high, low):
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - data)) / rng)
        
    #Calculate the euclidean distance
    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return distance

    #Locate the nearest neighbors for a row, return a list with the nearest neighbors
    def get_neighbors(self, row):  
        self.row = row
        distances=[]*0
        count = 0
        row0 = self.x_test[row]
        for train_row in self.dataset:
            dist = KNN.euclidean_distance(row0, train_row)
            distances.append([dist,count,self.dataset2[count][5]])
            count = count + 1
        distances = sorted(distances, key=lambda x: x[0], reverse=False)
        distances.pop(0)
        neighbors = list()
        for i in range(self.k):
            neighbors.append(distances[i][2])
        return neighbors
    
    #Predict if PM_HIGH = 0 or 1 for array x_test
    def predict(self, x_test):
        y_pred = list()
        self.x_test = x_test
        self.x_test = KNN.scale(x_test,1,0)
        count=0
        for row in x_test:
            neighbors = KNN.get_neighbors(self, count)
            if (sum(neighbors)/ len(neighbors)) >= 0.5:
                y_pred.append(1.0)
            else: y_pred.append(0.0)
            count = count+1
        return y_pred

    #Calculate the accuracy for predicted values for PM_HIGH
    def score(self, y_test, y_pred):
        count1 = 0
        count2 = 0
        for i in y_test:
            if y_test[count2] == y_pred[count2]:
                count1 = count1 + 1
            count2 = count2 + 1
        return count1/count2

#Create a new KNN with K= 10. Fit the training data with beijing and 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
clf = KNN(10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf.score(y_test, y_pred)

#Calculate the mean accuracy for n random states
def meanAccuracy(X, y, k, n_folds, test_size):
        scores = list()
        for i in range(1, n_folds):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
            clf = KNN(k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores.append(clf.score(y_test, y_pred))
        return sum(scores)/len(scores)

#Calculate the mean accuracy for the training data
meanAccuracy(X,y,10,5,0.20)

neighbors = np.arange(1,200)
test_accuracy = np.empty(len(neighbors))

#Iteratively creating a classifier with the different k's in 'neighbor' and adding the cross-validated accuracies to a list.
for i, k in enumerate(neighbors):
    print(i)
    test_accuracy[i] = meanAccuracy(X_test,y_test,k,5,0.20)

#Plotting the different accuracies dependent on k's 
plt.title('Uniform k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.savefig('KNN.png', dpi=400)
plt.show()

#Drop columns with correlation 0.1 < for shanghai and guangzhou
shanghai_df = shanghai_df.drop(columns=['season', 'PRES', 'TEMP', 'precipitation', 'cbwd_NE'])
Xs = shanghai_df.drop(['PM_HIGH'],axis=1)
Xs = Xs.to_numpy()
ys = shanghai_df['PM_HIGH']
ys = ys.to_numpy()
shanghai_df.head()

guangzhou_df = guangzhou_df.drop(columns=['season', 'PRES', 'TEMP', 'precipitation', 'cbwd_NE'])
Xg = guangzhou_df.drop(['PM_HIGH'],axis=1)
Xg = Xg.to_numpy()
yg = guangzhou_df['PM_HIGH']
yg = yg.to_numpy()

#New model with optimal nearest neighbors, k = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
clf = KNN(20)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf.score(y_test, y_pred)

#Score for Shanghai
ys_pred = clf.predict(Xs)
print(clf.score(ys_pred, ys))

#Score for Guangzhou
yg_pred = clf.predict(Xg)
print(clf.score(yg_pred,yg))

