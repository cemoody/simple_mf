import os.path
import numpy as np
import pandas as pd
import requests
from zipfile import ZipFile

# Download, unzip and read in the dataset
name = 'data/ml-1m.zip'
base = 'data/'

if not os.path.exists(base):
    os.mkdir(base)

if not os.path.exists(name):
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    r = requests.get(url)
    with open(name, 'wb') as fh:
        fh.write(r.content)
    zip = ZipFile(name)
    zip.extractall(base)


# Create users dataframe
users = pd.read_csv(base + "/ml-1m/users.dat", delimiter='::', engine='python',
                    names=['user', 'gender', 'age', 'occupation', 'zipcode'])


# Create users dataframe
movies = pd.read_csv(base + "/ml-1m/movies.dat", delimiter='::', engine='python',
                    names=['id', 'title', 'genres_str'])


# Create dict of genres
movies['genres'] = movies['genres_str'].apply(lambda gs: {key: 1.0 for key in gs.split('|')})

# Create ratings dataframe
cols = ['user', 'item', 'rating', 'timestamp']
rat = pd.read_csv(base + '/ml-1m/ratings.dat', delimiter='::',
                  names=cols, engine='python')
# Is this rating the first rating ever for that user, or the nth?
rat['rank'] = rat.groupby("user")["timestamp"].rank(ascending=True)
# Make our numbers predictable
np.random.seed(42)
# Set 90% of dataset to training, 10% test
rat['is_train'] = np.random.random(len(rat)) < 0.90
rat.to_pickle('data/dataset.pd')

# Merge ratings & user features
df = rat.merge(users, on='user')
df = df.merge(movies, left_on='item', right_on='id')
df = df.sample(frac=1)
assert len(rat) == len(df)

# Compute cardinalities
n_features = df.user.max() + 1 + df.item.max() + 1
n_user = df.user.max() + 1
n_item = df.item.max() + 1
n_rank = df['rank'].max() + 1
n_occu = df['occupation'].max() + 1
print('n_item', n_item)
print('n_user', n_user)
print('n_featuers', n_features)
print('n_occu', n_occu)
print('n_rows', len(df))


def split(subset):
    feat_cols = ['user', 'item', 'rank', 'occupation']
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

np.savez("data/dataset.npz", train_x=train_x, train_y=train_y,
         train_xy=train_xy, test_x=test_x, test_y=test_y, test_xy=test_xy,
         train_dict=train_dict, test_dict=test_dict,
         n_user=n_user, n_item=n_item, n_ranks=n_rank, n_occu=n_occu)
