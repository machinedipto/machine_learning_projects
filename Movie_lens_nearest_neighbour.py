import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation


datafile = '/media/light/UbuntuDrive/Python_Code/Propython/Recommendation_Systems/ml-100k/u.data'
data = pd.read_csv(datafile, sep='\t', header=None, names=[
                   'userid', 'itemid', 'ratings', 'timestamp'])

movieinfoFile = '/media/light/UbuntuDrive/Python_Code/Propython/Recommendation_Systems/ml-100k/u.item'
movieInfo = pd.read_csv(movieinfoFile, sep='|', header=None,
                        index_col=False, names=['itemid', 'title'], usecols=[0, 1])

movieInfo

data = pd.merge(data, movieInfo, left_on='itemid', right_on='itemid')

data

data = pd.DataFrame.sort_values(data, ['userid', 'itemid'], ascending=[0, 1])

data

movies_per_user = data.userid.value_counts()
movies_per_user

users_per_movie = data.title.value_counts()

# function to find out the top N favourite movies from the userid


def favouriteMovies(activeuser, N):
    # 1. filter with the active userid
    # 2. sort the values with rating in descending order
    # 3, pick the top N rows

    topmovies = pd.DataFrame.sort_values(data[data.userid == activeuser], [
                                         'ratings'], ascending=False)[:N]
    return list(topmovies.title)


favouriteMovies(5, 3)

# next we need to start with the K nearest neighbour based approach , first we need to do start with create
# a user vector where each element of the vector will be for the user ratinngs for one moveie as there are
# 1600 odd movies so the vector will be consist 1600 where not rated movies will be considered as nan

user_item_rating_matrix = pd.pivot_table(data, values='ratings', index=[
                                         'userid'], columns=['itemid'])

user_item_rating_matrix

# next is find the similarity between the users
# we will be taking the correlation between the users


def similarity(user1, user2):
    # this line will normalizze the user1 so there is no bias term ignoring the nan sort_values
    user1 = np.array(user1) - np.nanmean(user1)
    user2 = np.array(user2) - np.nanmean(user2)  # same as above

    # next we need to find the movies where both of the users are rated
    common_item_ids = [i for i in range(
        len(user1)) if user1[i] > 0 and user2[i] > 0]

    # if both the user have no movie in common rated we can return 0 or if there are movies common in both user return the correlation among them
    if len(common_item_ids) == 0:
        return 0
    else:
        user1_list = [user1[i] for i in common_item_ids]
        user1_array = np.asarray(user1_list)
        user2_list = [user2[i] for i in common_item_ids]
        user2_array = np.asarray(user2_list)
        return correlation(user1_array, user2_array)

# using the above similarity function find the nearest neighbour of the active users


similarity(user_item_rating_matrix.loc[5], user_item_rating_matrix.loc[1])


user1 = np.array(user_item_rating_matrix.loc[5])

user1

user2 = np.array(user_item_rating_matrix.loc[10])

common_item_ids = [i for i in range(
    len(user1)) if user1[i] > 0 and user2[i] > 0]

user1_array = [user1[i] for i in common_item_ids]
user2_array = [user2[i] for i in common_item_ids]

user2[49]
user2[98]
user2_array
correlation(np.asarray(user1_array), np.asarray(user2_array))


def nearestNeigbourRatings(activeuser, K):
    # This function will find the k nearest neighbours of the active user, then with the active users rating predict the active users rating for the other movies_per_user
    # creating a smimlarity matrix where the index will be the users and value will be the similarity between the activeuser and all the user

    similarity_matrix = pd.DataFrame(
        index=user_item_rating_matrix.index, columns=['similarity'])

    for i in user_item_rating_matrix.index:
        similarity_matrix.loc[i] = similarity(
            user_item_rating_matrix.loc[activeuser], user_item_rating_matrix.loc[i])

    # now after creating the matrix sort the values with similarity values in desceding order and get the first k sort_values

    similarity_matrix = pd.DataFrame.sort_values(
        similarity_matrix, ['similarity'], ascending=False)

    nearest_neighbour = similarity_matrix[:K]

    # after finding out the K nearest neighbour find their rating to actually predict the active user ratings
    nearest_neigbour_ratings = user_item_rating_matrix.loc[nearest_neighbour.index]

    # creating a new dataframe where all the predicted
    predict_item_rating = pd.DataFrame(
        index=user_item_rating_matrix.index, columns=['predicted_rating'])

    for i in user_item_rating_matrix.columns:
        ave_predict = np.nanmean(user_item_rating_matrix.loc[activeuser])

        for j in nearest_neighbour.index:
            predictedRating = ave_predict + (user_item_rating_matrix.loc[j, i]
                                             - np.nanmean(user_item_rating_matrix.loc[j])) * nearest_neighbour.loc[j, 'similarity']

        predict_item_rating.loc[i, 'predicted_rating'] = predictedRating

    return predict_item_rating


def topNrecommendation(activeuser, N):
    predict_item_rating = nearestNeigbourRatings(activeuser, 10)

    movies_already_watched = list(
        user_item_rating_matrix.loc[activeuser].loc[user_item_rating_matrix.loc[activeuser] > 0].index)
    predict_item_rating_after_drop = predict_item_rating.drop(
        movies_already_watched)

    toprecommendation = pd.DataFrame.sort_values(predict_item_rating_after_drop, [
                                                 'predicted_rating'], ascending=False)[:N]

    toprecommendationtitles = (
        movieInfo.loc[movieInfo.itemid.isin(toprecommendation.index)])

    return list(toprecommendationtitles.title)


activeuser = 3

print(favouriteMovies(activeuser, 5), '\n', topNrecommendation(activeuser, 5))


####################LATENT FACTOR BASED METHOD ###################


def matrixFactorization(R, K, steps=10, gamma=0.001, lamda=0.02):
    # R is the user item rating matrix
    # K is the number of factors we will find
    # we will be using stochastic gradient descent to find the factor vectors

    N = len(R.index)
    M = len(R.columns)

    p = pd.DataFrame(np.random.rand(N, K), index=R.index)
    # p is here user factor matrix we inititlized with some random values
    q = pd.DataFrame(np.random.rand(M, K), index=R.columns)
    # q is the product factor matrix

    # now implementing SGD
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    eij = R.loc[i, j] - np.dot(p.loc[i], q.loc[j])
                    # this is the error value which is the difference between actual rated value minus the factored rate
                    # we have to minimize the error fuunction
                    p.loc[i] = p.loc[i] + gamma * \
                        (eij * q.loc[j] - lamda * p.loc[i])
                    # the p and q need to be moved in the downward direction
                    # gamma is the step taken towards slope and lambda is the regularization parameter
                    q.loc[j] = q.loc[j] + gamma * \
                        (eij * p.loc[i] - lamda * q.loc[j])

        # At the end of this process we have looped through the entire data once now we need to check the error is brought down to least or not
        e = 0
        for i in R.index:
            for j in R.columns:
                if R.loc[i, j] > 0:
                    # sum of sqaures for the error term
                    e = e + pow(R.loc[i, j] - np.dot(p.loc[i], q.loc[j]), 2) + lamda * (
                        pow(np.linalg.norm(p.loc[i]), 2) + pow(np.linalg.norm(q.loc[j]), 2))

        if e < 0.001:
            break
        print(step)
    return p, q


(P, Q) = matrixFactorization(user_item_rating_matrix,
                             K=2, gamma=0.001, lamda=0.02, steps=100)

predict_item_rating_latent = pd.DataFrame(
    np.dot(P.loc[activeuser], Q.T), index=Q.index, columns=["rating"])

predict_item_rating_latent
movies_already_watched = list(
    user_item_rating_matrix.loc[activeuser].loc[user_item_rating_matrix.loc[activeuser] > 0].index)
predict_item_rating_after_drop = predict_item_rating_latent.drop(
    movies_already_watched)

toprecommendation = pd.DataFrame.sort_values(predict_item_rating_after_drop, [
                                             'rating'], ascending=False)[:5]
toprecommendationtitles = movieInfo.loc[movieInfo.itemid.isin(
    toprecommendation.index)]

print(favouriteMovies(activeuser, 5), toprecommendationtitles.title)
