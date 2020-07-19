import os.path
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile

# Download, unzip and read in the dataset
name = 'data/ml-20m.zip'
base = 'data/'

if not os.path.exists(base):
    os.mkdir(base)

if not os.path.exists(name):
    url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
    r = requests.get(url)
    with open(name, 'wb') as fh:
        fh.write(r.content)
    zip = ZipFile(name)
    zip.extractall(base)


# Create movies dataframe
movies = pd.read_csv(base + "/ml-20m/movies.csv", delimiter=',', engine='python',
                    names=['id', 'title', 'genres_str'], header=1)


# Create dict of genres
movies['genres'] = movies['genres_str'].apply(lambda gs: {key: 1.0 for key in gs.split('|')})

# Create ratings dataframe
cols = ['user', 'item', 'rating', 'timestamp']
rat = pd.read_csv(base + '/ml-20m/ratings.csv', delimiter=',',
                  names=cols, engine='python', header=1)
# Is this rating the first rating ever for that user, or the nth?
rat['rank'] = rat.groupby("user")["timestamp"].rank(ascending=True, method='first').astype(int) - 1
# Make our numbers predictable
np.random.seed(42)
# Set 90% of dataset to training, 10% test
rat['is_train'] = rat.user.apply(lambda x: hash(x) % 10 != 0)


# Merge ratings & user features
df = rat.merge(movies, left_on='item', right_on='id')
df = df.sample(frac=1)
# assert len(rat) == len(df)

#  Create rating-10 field so that each rating
# becomes an integer. Originally, ratings are  [0.5, 1.0, 1.5, ...5.0]
df['rating10']  = (df['rating'] * 2).astype(int)

# Compute cardinalities
n_features = df.user.max() + 1 + df.item.max() + 1
n_user = df.user.max() + 1
n_item = df.item.max() + 1
n_rank = df['rank'].max() + 1
print('n_item', n_item)
print('n_user', n_user)
print('n_features', n_features)
print('n_rows', len(df))


def split(subset):
    feat_cols = ['user', 'item', 'rank']
    out_cols = ['rating']
    features = subset[feat_cols]
    features_dict = list(subset['genres'].values)
    outcomes = subset[out_cols]
    features = features.values.astype(np.int32)
    outcomes = outcomes.values.astype(np.float32)
    both = subset[feat_cols + out_cols]
    return features, outcomes, both, features_dict


train_x, train_y, train_xy, train_dict = split(df[df.is_train])
test_x, test_y, test_xy, test_dict = split(df[~df.is_train])

# np.savez("data/dataset_ml20.npz", train_x=train_x, train_y=train_y,
#          train_xy=train_xy, test_x=test_x, test_y=test_y, test_xy=test_xy,
#          train_dict=train_dict, test_dict=test_dict,
#          n_user=n_user, n_item=n_item, n_ranks=n_rank)

def split_stream(sub):
    items = np.zeros((n_user, 1000), dtype='int32')
    ratng = np.zeros((n_user, 1000), dtype='int32')
    users = np.zeros(n_user, dtype='int32')
    users[sub['user'].values] = sub['user'].values
    items[sub['user'].values, sub['rank'].values] = sub['item'].values
    ratng[sub['user'].values, sub['rank'].values] = sub['rating10'].values
    idx = ~np.all(items == 0, axis=1)
    return items[idx, :], ratng[idx, :], users[idx]


# Only keep  first 1k  ratings for every  user
sub = df[df['rank'] < 1000]
train_items, train_ratng, train_user = split_stream(sub[sub.is_train])
test_items, test_ratng, test_user = split_stream(sub[~sub.is_train])

np.savez("data/dataset_ml20_wide.npz", 
         train_items=train_items, train_ratng=train_ratng, train_user=train_user,
         test_items=test_items, test_ratng=test_ratng, test_user=test_user)