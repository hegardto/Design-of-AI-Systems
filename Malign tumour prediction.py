# -*- coding: utf-8 -*-

# -- Sheet --

# Eickhoff (2016) (Links to an external site.) summarizes these as follows: "The size of cells tends to be homogeneous given a speciﬁc type of tissue. 1) The presence of signiﬁcantly larger cells is evidence for the uncontrolled growth that is indicative of malignant tumors. 2) The shape of benign cells usually shows only limited variance, whereas malignant cells can develop arbitrary structures that do not conform with the general pattern of their surroundings. 3) The color of the cell nucleus should be identical for regular cells of the same type. Cancer cells often have signiﬁcantly larger and darker nuclei that are more densely packed with DNA. 4) Regular cells show similar texture. Malignant tumors, on the other hand, can range from smooth surfaces to ragged or lumpy textures for neighbouring cells. 5) Finally, for healthy tissue, cell arrangement tends to be orderly, with regular distances between cells. Cancer cells can spread out or clutter almost arbitrarily.


# Implement and evaluate the following classifiers in the task of assigning the diagnosis malignant or benign:
# A rule-based classifier which follows the following form:
# - If [cell size is abnormal]:
# - or [cell shape is abnormal]
# - or [cell texture is abnormal]
# - or [cell arrangement/similarity/homogeneity is abnormal],
# - then: diagnosis is malignant, 
# - otherwise: diagnosis is benign.


import pickle
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with open('wdbc.pkl', 'rb') as f:
    data = pickle.load(f)

def scale(data, high, low):
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - data)) / rng)

data.columns = data.columns.str.replace(' ', '_')

data2 = data.drop(data[data.malignant == 1].index)

data3 = data.drop(data[data.malignant == 0].index)

normalized = scale(data, 1, 0)

normalized2 = normalized.drop(normalized[normalized.malignant == 1].index)

normalized3 = normalized.drop(normalized[normalized.malignant == 0].index)

data['homogenity_0'] = normalized['area_1'] + normalized['radius_1'] + normalized['perimeter_1']
data2['homogenity_0'] = normalized2['area_1'] + normalized2['radius_1'] + normalized2['perimeter_1']
data3['homogenity_0'] = normalized3['area_1'] + normalized3['radius_1'] + normalized3['perimeter_1']
data['homogenity_1'] = data['homogenity_0'].std()

data

corrdata = data[['malignant','radius_0','texture_0','perimeter_0','area_0','smoothness_0','compactness_0','concavity_0','concave_points_0','symmetry_0','fractal_dimension_0']]

#Calculate correlations between all features in dataset
corrmat = corrdata.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))

#Plot heat map
g=sns.heatmap(corrdata[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.savefig('heatmap', dpi=400)

corrdata = data[['malignant','radius_1','texture_1','perimeter_1','area_1','smoothness_1','compactness_1','concavity_1','concave_points_1','symmetry_1','fractal_dimension_1']]

#Calculate correlations between all features in dataset
corrmat = corrdata.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#Plot heat map
g=sns.heatmap(corrdata[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.savefig('heatmap2',dpi=400)

area_mean = data2['area_0'].mean()
area_sd = data2['area_1'].sum()/math.sqrt(len(data2))

conc_mean = data2['concave_points_0'].mean()
conc_sd = data2['concave_points_1'].sum()/math.sqrt(len(data2))

smooth_mean = data2['smoothness_0'].mean()
smooth_sd = data2['smoothness_1'].sum()/math.sqrt(len(data2))

texture_mean = data2['texture_0'].mean()
texture_sd = data2['texture_1'].sum()/math.sqrt(len(data2))

fractal_mean = data2['fractal_dimension_0'].mean()
fractal_sd = data2['fractal_dimension_1'].sum()/math.sqrt(len(data2))

homogenity_mean = data['homogenity_0'].mean()
homogenity_sd = data['homogenity_1'].sum()/math.sqrt(len(data))

malign = []
benign = []
nr_of_sds = 0.59

#Ska mean räknas på alla patienter eller endast på de friska?
for index, row in data.iterrows():
    if row['area_0'] > 685:
        malign.append(row['id'])
    elif row['concave_points_0'] > 0.05 and row['smoothness_0'] > 0.09:
        malign.append(row['id'])
    elif row['texture_0'] > 30:
        malign.append(row['id'])
    elif row['homogenity_0'] > 0.40:
        malign.append(row['id'])
    else:
        benign.append(row['id'])

print (malign)
print (len(malign))
print (len(malign+benign))

count1 = 0
count2 = 0

for index, row in data.iterrows():
    for patient in malign:
        if row['id'] == patient and row['malignant'] == 1:
            count1 += 1
    for patient in benign:
        if row['id'] == patient and row['malignant'] == 0:
            count1 += 1

print ("Accuracy " + str(count1/(len(data))))

import numpy
from matplotlib import pyplot

#Area
x = [data2['area_0']]
y = [data3['area_0']]

bins = numpy.linspace(0, 1500, 100)
plt.figure(figsize=(20,10))
plt.title('Mean sample area', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


pyplot.hist(x, bins, alpha=0.5, label='Benign')
pyplot.hist(y, bins, alpha=0.5, label='Malign')
pyplot.legend(loc='upper right')
pyplot.grid()
plt.savefig('area', dpi=400)
pyplot.show()

#Concavity
x = [data2['concave_points_0']]
y = [data3['concave_points_0']]

bins = numpy.linspace(0, 0.2, 100)
plt.figure(figsize=(20,10))
plt.title('Mean sample concave points', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


pyplot.hist(x, bins, alpha=0.5, label='Benign')
pyplot.hist(y, bins, alpha=0.5, label='Malign')
pyplot.legend(loc='upper right')
pyplot.grid()
plt.savefig('concave points', dpi=400)
pyplot.show()

#Smoothness
x = [data2['smoothness_0']]
y = [data3['smoothness_0']]

bins = numpy.linspace(0.05, 0.15, 100)
plt.figure(figsize=(20,10))
plt.title('Mean sample smoothness', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

pyplot.hist(x, bins, alpha=0.5, label='Benign')
pyplot.hist(y, bins, alpha=0.5, label='Malign')
pyplot.legend(loc='upper right')
pyplot.grid()
plt.savefig('smoothness', dpi=400)
pyplot.show()

#Texture
x = [data2['texture_0']]
y = [data3['texture_0']]

bins = numpy.linspace(0, 40, 100)
plt.figure(figsize=(20,10))
plt.title('Mean sample texture', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

pyplot.hist(x, bins, alpha=0.5, label='Benign')
pyplot.hist(y, bins, alpha=0.5, label='Malign')
pyplot.legend(loc='upper right')
pyplot.grid()
plt.savefig('texture', dpi=400)
pyplot.show()

#Symmetry
x = [data2['homogenity_0']]
y = [data3['homogenity_0']]

bins = numpy.linspace(0, 1.5, 100)
plt.figure(figsize=(20,10))
plt.title('Mean sample homogenity', fontsize=30, fontweight='bold')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

pyplot.hist(x, bins, alpha=0.5, label='Benign')
pyplot.hist(y, bins, alpha=0.5, label='Malign')
pyplot.legend(loc='upper right')
pyplot.grid()
plt.savefig('homogenity', dpi=400)
pyplot.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

y = data['malignant']
X = data.drop('malignant', 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.25)

#Create a Random Forest Classifier
#n_estimators is the number of trees in the forest
clf = RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets
clf.fit(X_train,y_train)

# Make predictions
y_pred=clf.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Model Accuracy
score = cross_val_score(clf, X, y, cv=10)
print(score.mean())

from sklearn import tree
from IPython.display import Image  
import pydotplus # Maybe you need to pip install this package?'
#from graphviz import Digraph

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
y_pred = clf.predict(X_test) #protesterar

print(accuracy_score(y_test, y_pred))

score = cross_val_score(clf, X, y, cv=10)
print(score.mean())

# Create DOT data
dot_data = tree.export_graphviz(clf, out_file=None, rounded = True, proportion = False, precision = 2, filled = True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  


# Show graph
Image(graph.create_png())

graph.write_png("iris.png")

