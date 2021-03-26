# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np

Genres = pd.read_csv ('movie_genres.csv')
Reviews = pd.read_csv('user_reviews.csv')
del Reviews['Unnamed: 0']

Genres

#spara
#dfR2 = dfR.transpose()
#dfR2 = dfR2.drop(['Unnamed: 0'])
#header_row = 0
#dfR2.columns = dfR2.iloc[header_row]
#dfR2 = dfR2.drop(['User'])
#dfG.reset_index(drop=True, inplace=True)
#dfR2.reset_index(drop=True, inplace=True)
#result = pd.concat([dfG, dfR2], axis=1, join="inner")
#result = result.drop(['Unnamed: 0'], axis=1)
#pd.set_option('max_columns', None)
#result.head()

Genres_df1 = Genres.transpose()
Genres_df1 = Genres_df1.drop(['Unnamed: 0'])
header_row = 0
Genres_df1.columns = Genres_df1.iloc[header_row]
Genres_df1 = Genres_df1.drop(['movie_title'])

Reviews_df1 = Reviews.drop(['Unnamed: 0', 'User'], axis=1)
Genres_df1.head()

# 


Reviews_df1.head()

from surprise import SVD
from surprise import Dataset
from surprise import Reader

algo = SVD()
reader = Reader(line_format='item user rating', sep=',', rating_scale=(1,5), skip_lines=1)
data = Dataset.load_from_file('user_reviews.csv',reader)
trainset = data.build_full_trainset()
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)
predictions

from surprise.model_selection import cross_validate
cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=True)

from collections import defaultdict

def get_top_5(predictions):

    # First map the predictions to each user.
    top_5 = defaultdict(list)
    for user, movie, true_r, est, _ in predictions:
        top_5[user].append((movie, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for user, user_ratings in top_5.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_5[user] = user_ratings[:5]

    return top_5

top_5 = get_top_5(predictions)

# Print the recommended items for a chosen user
Reviews_df1 = Reviews.drop('User', axis=1)
def top5(user):
    list1 = []
    for uid, user_ratings in top_5.items():
        if uid==user:
            list1.append([iid for (iid, _) in user_ratings])
    list2 = []
    for obj in list1:
        for o in obj:   
            list2.append(Reviews_df1.columns[int(o)])
    print (user + " should really watch the following 5 movies:")
    for o in list2:
        print(o)

top5('Mariana')

# Example of calculating Euclidean distance
from math import sqrt

# calculate the Euclidean distance between two rows
def euclidean_distance(row1, row2):
	distance = 0.0
	count = 0.0
	for i in range(len(row1)-1):
		if row1[i] != 0 and row2[i] !=0:
			distance += (row1[i] - row2[i])**2
			count = count+1
	if count == 0:
		return sqrt(distance)
	return (sqrt(distance)/count)

dataset = Reviews.to_numpy()
dataset2 = np.delete(dataset,0, 1)
row0 = dataset2[4]
arr=[]*0

for row in dataset2:
	distance = euclidean_distance(row0, row)
	arr.append(distance)

du = pd.DataFrame(arr, columns = ['Distance']) 
du['User'] = Reviews['User']
du = du[du['Distance'] != 0]
du = du.append({'User': 'Javier', 'Distance': 0} , ignore_index=True)

du = du.sort_values(by=['Distance'], ascending=True)
du

#Create a list with the index number from dataframe Reviews for 5 nearest neighbours 
nearest_neighbors= []

for i in range(0,5):
    nearest_neighbors.append(du.index[i])
nearest_neighbors

#plocka ut alla filmer som ace har sett som inte Vincent sett
#ReviewsDropped = Reviews['User'] = 'Ace'


Satan  = Reviews[Reviews.index.isin(nearest_neighbors)]
#dfu= Filter_df.join(du)

#Create reviews dataframe with nearest neighbors
Filter_df = pd.merge(du, Satan, left_index=True, right_index=True)

#Drop all movies that X has already seen
Filter_df = Filter_df.loc[:, (Filter_df != 0).any(axis=0)]
Filter_df = Filter_df.loc[: ,(Filter_df == 0).any(axis=0)]


df_new =  Filter_df[Filter_df.columns[Filter_df.iloc[0] == 0.0]]
df_new.mean(axis = 0) 

df_new = df_new.sort_values(by = 'Distance', ascending=True) 


df_new = df_new.replace(0, np.NaN)
df_new.loc['Mean'] = df_new.mean()
rslt_df = df_new.sort_values(by = 'Mean', axis = 1, ascending=False) 

recomendas = rslt_df.iloc[:, 1:6]

print('\033[1m' + 'The top 5 recommendations for ' + du['User'].iloc[0] + '\033[0m')
for col in recomendas.columns: 
    print(col) 

