
# coding: utf-8

# # CPSC 340 a5 q3: recommender systems
# 

# In this exercise we'll be exploring movie recommendations using the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. We'll use the small version of the data set which you can download [here](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip). **Before proceeding, please download it and put the unzipped `ml-latest-small` directory inside your `data` directory.** The structure of the data is described in the [README](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html) that comes with the data. 
# 
# Dependencies: you'll need the Pandas package for this question. If you're using Anaconda, you'll already have it. Otherwise you should be able to get it with `pip install pandas`.

# In[1]:

import pickle
import os
import numpy as np
import numpy.linalg as npla
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
get_ipython().magic('matplotlib inline')


# ### Introducing the ratings data
# 
# Let's start by looking at the data.

# In[2]:

ratings = pd.read_csv(os.path.join("..", "data", "ml-latest-small", "ratings.csv"))
ratings.head(10)


# As we can see, the `ratings` DataFrame contains one row per rating, which tells us the `userId` of the person giving the rating, the `movieId` of the movie being rating, and the rating itself out of 5 stars.
# 
# The next block of code does some preprocessing and prints out some key numbers...

# In[3]:

N = len(np.unique(ratings["userId"]))
M = len(np.unique(ratings["movieId"]))

# since the id values aren't contiguous, we need a mapping from id to index of an array
N_mapper = dict(zip(np.unique(ratings["userId"]), list(range(N))))
M_mapper = dict(zip(np.unique(ratings["movieId"]), list(range(M))))

print("Number of users (N)                : %d" % N)
print("Number of movies (M)               : %d" % M)
print("Number of ratings (|R|)            : %d" % len(ratings))
print("Fraction of nonzero elements in Y  : %.1f%%" % (len(ratings)/(N*M)*100))
print("Average number of ratings per user : %.0f" % (len(ratings)/N))
print("Average number of ratings per movie: %.0f" % (len(ratings)/M))


# Next, let's split `ratings` into a training and validation set:

# In[4]:

train_ratings, valid_ratings = train_test_split(ratings, test_size=0.2, random_state=42)


# Let's now construct $Y$, which is defined above, from the `ratings` DataFrame. 

# In[5]:

def create_Y_from_ratings(ratings_df, N, M):    
    Y = np.zeros((N,M)) 
    Y.fill(np.nan)
    for index, val in ratings_df.iterrows():
        n = N_mapper[val["userId"]]
        m = M_mapper[val["movieId"]]
        Y[n,m] = val["rating"]
    
    return Y

Y          = create_Y_from_ratings(train_ratings, N, M)
Y_validate = create_Y_from_ratings(valid_ratings, N, M)


# Above we committed a mortal sin, which is storing `Y` as a dense numpy array. If we had more data, we would need to use a [sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html) data type, which we've perhaps mentioned very briefly. But for now we won't worry about it. 
# 
# Also, for convenience, we store the missing entries as `NaN` instead of zero. The reason will become apparent soon.

# In[6]:

print("Number of NaNs we are storing in `Y` because Mike is sloppy: %.1e" % (N*M-len(ratings)))


# ### Introducting the notation
# 
# Here is some notation we will be using. This is different from what we have been doing in class (sorry).
# 
# **Constants**:
# 
#  - $N$: the number of users, indexed by $n$.
#  - $M$: the number of movies, indexed by $m$.
#  - $d$: the nubmer of movie features (more on this later).
#  - $k$: the number of latent dimensions we use (more on this later).
#  - $\mathcal{R}$: the set of indices $(n,m)$ where we have ratings in $Y$ (so $|\mathcal{R}|$ is the total number of ratings).
#  
# **The data**:
# 
#  - $Y$: the matrix containing the ratings (size $N\times M$), with a lot of missing entries. $y_{nm}$ is one rating.
#  - $Z$: a matrix whose rows $z_m$ represent the features for movie $m$ (size $M\times d$).
#  
# **Learned parameters** (more on these later):
# 
#  - $b_n$: a bias variable specific to user $n$.
#  - $b_m$: a bias variable specific to movie $m$.
#  - $U$: a matrix whose rows $u_n^T$ represent latent features for user $n$ (size $N \times k$).
#  - $V$ : a matrix whose columns $v_m$ represent latent features for movie $m$ (size $k \times M$).  
#  - $w$: the weight vector for linear regression on the movie features (length $d$).
#  - $w_n$: the same as $w$ but separate for each user
# 

# ### Introducing the features
# 
# Later on we'll try to use some features or "context" to help us make recommendations. We'll just use the genres of the movies although these aren't particularly great features. We'll store the features in a matrix called $Z$, which has size $M\times d$.

# In[7]:

movies = pd.read_csv(os.path.join("..", "data", "ml-latest-small", "movies.csv"))
movies["genres"] = movies["genres"].apply(lambda x: x.split("|"))
movies.head()


# The `movies` DataFrame, loaded above, contains the movie titles and genres.
# 
# We'll start by just using the genres, with binary features representing the presence/absense of a particular genre. If you want, later on you can consider using other features like the year or even title. 

# In[8]:

genres_set = set(g for G in movies["genres"] for g in G)
d = len(genres_set) 
print("We have %d genres and thus %d (binary) movie features.\n" % (d,d))
print("Here they are:\n\n * %s\n" % "\n * ".join(genres_set))


# By the way, `(no genres listed)` is left in as a feature. You could remove this and have movies with no genre have the zero vectror as their feature vector. This would make $d=19$ instead of $d=20$. I'm not sure it matters much. 
# 
# We now preprocess the features to get them into our `Z` matrix. Again, this should probably be a sparse matrix.

# In[9]:

# preprocess the features
genres_dict = {g:i for i,g in enumerate(genres_set)}
Z = np.zeros((M,d))
for index, val in movies.iterrows():
    if val["movieId"] not in M_mapper: # movie wasn't rated (but I thought those weren't supposed to be included??)
        continue
    m = M_mapper[val["movieId"]]
    for g in val["genres"]: 
        Z[m,genres_dict[g]] = 1

print("Average number of genres per movie: %.1f" % (np.sum(Z)/M))


# By the way, if you check out the [MovieLens](https://grouplens.org/datasets/movielens/) page you'll see there's a bigger version of the data set that includes "tag genome" data, which can basically be used as more features. I wrote some code to preprocess these features but am not including it here as I think there's enough going on. If you are interested, you could try that, but it involves quite a bit of data wrangling -- you probably won't have time until after the course ends.

# ### Introducing the models
# 
# Here are the models we'll consider for our recommender system:
# 
# 1. global average rating
# 2. user average rating
# 3. movie average rating
# 4. average of (2) and (3) above
# 5. linear regression on movie features, globally
# 6. linear regression on movie features, separately for each user
# 7. SVD (naively treating missing entries as 0)
# 8. SVD (treating missing entries as missing, via gradient descent)
# 9. Combining (8) with (6)
# 10. Same as (9) but trained using SGD instead of GD
# 
# Roughly speaking, we are going to be learning models that look like
# 
# $$\hat{y}_{nm} = \frac{b_u + b_m}{2} + u_n^T v_m + w_n^T z_m$$
# 
# The model above in particular corresponds to model (9) above. Take your time to digest this before proceeding. You may need to refer back to the notation above. I know you're used to $w$ and $z$ being the latent factors and factor loadings, but I'm using $v$ and $u$ for those since I need $w$ and $z$ to serve a different purpose.

# ### Our loss function
# 
# For all approaches we will measure performance with mean squared error on the validation set, which means that our error for a particular set of predictions $\hat{y}_{nm}$ is given by
# 
# $$ f(\textrm{parameters})= \frac{1}{|\mathcal{R}|} \sum_{(n,m)\in\mathcal{R}} (y_{nm} − \hat{y}_{nm})^2 $$
# 
# where $y_{nm}$ is the true rating and $\hat{y}_{nm}$ is the predicted rating.
# 
# The function below will compute this score for us. The `nanmean` function takes the mean of all elements but ignores the NaN values. This is why we set up $Y$ to have the missing enties as `NaN` instead of zero -- it's just very convenient now.

# In[10]:

def score(Y1, Y2):
    return np.nanmean( (Y1-Y2)**2 )


# ### The experiments: methods 1-9

# ** 1. Global average **

# In[11]:

avg = np.nanmean(Y)
Y_pred_1 = np.zeros(Y.shape) + avg
print("Global average train loss: %f" % score(Y_pred_1, Y))
print("Global average valid loss: %f" % score(Y_pred_1, Y_validate))


# ** 2. Per-user average**

# In[12]:

avg_n = np.nanmean(Y,axis=1)
avg_n[np.isnan(avg_n)] = avg
Y_pred_2 = np.tile(avg_n[:,None], (1,M))
print("Per-user average train loss: %f" % score(Y_pred_2, Y))
print("Per-user average valid loss: %f" % score(Y_pred_2, Y_validate))


# ** 3. Per-movie average **

# In[13]:

avg_m = np.nanmean(Y,axis=0)
avg_m[np.isnan(avg_m)] = avg # if you have zero ratings for a movie, use global average
Y_pred_3 = np.tile(avg_m[None,:], (N,1))
print("Per-movie average train loss: %f" % score(Y_pred_3, Y))
print("Per-movie average valid loss: %f" % score(Y_pred_3, Y_validate))


# ** 4. Average of per-user and per-movie averages **

# In[14]:

Y_pred_4 = 0 # TODO: YOUR CODE HERE

print("Per-movie average train loss: %f" % score(Y_pred_4, Y))
print("Per-movie average valid loss: %f" % score(Y_pred_4, Y_validate))


# ** 5. Linear regression with movie features **
# 
# Note: in this model we predict the same thing for each movie, regardless of the user, like in (3)

# In[15]:

# take training set ratings and put them in a vector
def get_lr_data(ratings_df, d):
    lr_y = np.zeros(len(ratings_df))
    lr_X = np.zeros((len(ratings_df), d))
    i=0
    for index, val in ratings_df.iterrows():
        m = M_mapper[val["movieId"]]
        lr_X[i] = Z[m]
        lr_y[i] = val["rating"]
        i += 1
    return lr_X, lr_y


# In[16]:

lr_features_train, lr_targets_train = get_lr_data(train_ratings, d)
lr_features_valid, lr_targets_valid = get_lr_data(valid_ratings, d)


# In[17]:

lr = LinearRegression()
lr.fit(lr_features_train, lr_targets_train)
Y_pred_5 = np.tile(lr.predict(Z), (N,1))
print("Genre features train loss: %f" % score(Y_pred_5, Y))
print("Genre features valid loss: %f" % score(Y_pred_5, Y_validate))


# ** 6 Per-user linear regressions on genre **
# 
# Below we do the preprocessing for you. But you'll probably need to read through the preprocessing code and understand it in order to finish the job.

# In[18]:

from collections import defaultdict
def get_lr_data_per_user(ratings_df, d):
    lr_y = defaultdict(list)
    lr_X = defaultdict(list)

    for index, val in ratings_df.iterrows():
        n = N_mapper[val["userId"]]
        m = M_mapper[val["movieId"]]
        lr_X[n].append(Z[m])
        lr_y[n].append(val["rating"])

    for n in lr_X:
        lr_X[n] = np.array(lr_X[n])
        lr_y[n] = np.array(lr_y[n])
        
    return lr_X, lr_y


# In[19]:

lr_featres_train_usr, lr_targets_train_usr = get_lr_data_per_user(train_ratings, d)
lr_featres_valid_usr, lr_targets_valid_usr = get_lr_data_per_user(valid_ratings, d)


# In[21]:

Y_pred_6 = 0 # TODO: YOUR CODE HERE

print("Per-user genre features train loss: %f" % score(Y_pred_6, Y))
print("Per-user genre features valid loss: %f" % score(Y_pred_6, Y_validate))


# ** 7. SVD with per-user and per-movie averages **
# 
# (It would probably be a good idea to use [sparse SVD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html), but I'm not doing it).
# 

# In[22]:

def predict_svd(U,V,b_n,b_m):
    return U@V + 0.5*b_n[:,None] + 0.5*b_m[None]


# In[23]:

k = 10 # defined above, the number of latent dimensions

# prepare data
Y_svd = Y - 0.5*avg_n[:,None] - 0.5*avg_m[None]
Y_svd[np.isnan(Y_svd)] = 0

svd = TruncatedSVD(n_components=k)
U = svd.fit_transform(Y_svd)
V = svd.components_
Y_pred_7 = predict_svd(U,V,avg_n,avg_m)
print("SVD train loss: %f" % score(Y_pred_7, Y))
print("SVD valid loss: %f" % score(Y_pred_7, Y_validate))


# ** 8. SVD with a proper handling of missing features **
# 
# We use gradient descent to fit. We implement the gradient calculations a bit weirdly to take advantage of code vectorization.

# In[24]:

# Initialize parameters
# - for the biases, we'll use the user/item averages
# - for the latent factors, we'll use small random values
b_n = avg_n.copy()
b_m = avg_m.copy()
V = 1e-5*np.random.randn(k,M)
U = 1e-5*np.random.randn(N,k)

# Optimization
nIter = 501
alpha = 0.0005

for itera in range(nIter):

    # Compute loss function value, for user's information
    if itera % 100 == 0:
        Ypred = predict_svd(U,V,b_n,b_m)
        train_loss = score(Ypred, Y)
        valid_loss = score(Ypred, Y_validate)
        print('Iter = %03d, train = %f, valid = %f'%(itera,train_loss,valid_loss))

    # Compute gradients
    Yhat = predict_svd(U,V,b_n,b_m)
    r = Yhat - Y
    r[np.isnan(r)] = 0
    g_b_n = 0.5*np.sum(r,axis=1)
    g_b_m = 0.5*np.sum(r,axis=0)
    g_V = U.T@r
    g_U = r@V.T
    
    # Take a small step in the negative gradient directions
    b_n -= alpha*g_b_n
    b_m -= alpha*g_b_m
    V -= alpha*g_V
    U -= alpha*g_U
    
Y_pred_8 = predict_svd(U,V,b_n,b_m)
print()
print("SVD GD train loss: %f" % score(Y_pred_8, Y))
print("SVD GD valid loss: %f" % score(Y_pred_8, Y_validate))


# ** 9. Gradient descent plus per-user movie features **

# In[25]:

# Initialize parameters
# - for the biases, we'll use the user/item averages
# - for the latent factors, we'll use small random values
b_n = avg_n.copy()
b_m = avg_m.copy()
V = 1e-5*np.random.randn(k,M)
U = 1e-5*np.random.randn(N,k)
W = 1e-5*np.random.randn(d,N)

# Optimization
nIter = 501
alpha = 0.0005

for itera in range(nIter):
    
    # Compute loss function value, for user's information
    if itera % 100 == 0:
        Ypred = predict_svd(U,V,b_n,b_m) + (Z@W).T
        train_loss = score(Ypred, Y)
        valid_loss = score(Ypred, Y_validate)
        print('Iter = %03d, train = %f, valid = %f'%(itera,train_loss,valid_loss))

    Yhat = predict_svd(U,V,b_n,b_m) + (Z@W).T
    r = Yhat - Y
    r[np.isnan(r)] = 0
    g_b_n = 0.5*np.sum(r,axis=1)
    g_b_m = 0.5*np.sum(r,axis=0)
    g_V = U.T@r
    g_U = r@V.T
    g_W = (r@Z).T
    
    # Take a small step in the negative gradient directions
    b_n -= alpha*g_b_n
    b_m -= alpha*g_b_m
    V -= alpha*g_V
    U -= alpha*g_U
    W -= alpha*g_W


# In[26]:

Y_pred_9 = 0 # TODO: make predictions given the trained model

print("Per-movie average train loss: %f" % score(Y_pred_9, Y))
print("Per-movie average valid loss: %f" % score(Y_pred_9, Y_validate))


# ### Compare the different methods

# In[28]:

methods = np.arange(1,10)

train = [score(eval("Y_pred_%d"%i), Y) for i in methods]
valid = [score(eval("Y_pred_%d"%i), Y_validate) for i in methods]


# In[29]:

pd.options.display.float_format = '{:,.2f}'.format # make things look prettier when printing
df = pd.DataFrame.from_dict({"training MSE": train, "validation MSE" : valid})
df.index = methods
df.T


# In[ ]:



