import os.path
import numpy as np
import requests
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

# Download, unzip and read in the dataset
name = 'data/ml-1m.zip'
base = 'data/ml-1m'
if not os.path.exists(name):
    url = 'http://files.grouplens.org/datasets/movielens/' + name
    r = requests.get(url)
    with open(name, 'wb') as fh:
        fh.write(r.content)
    zip = ZipFile(name)
    zip.extractall()

# First col is user, 2nd is movie id, 3rd is rating
data = np.genfromtxt(base + '/ratings.dat', delimiter='::')
# print("WARNING: Subsetting data")
# data = data[::100, :]
user = data[:, 0].astype('int32')
movie = data[:, 1].astype('int32')
rating = data[:, 2].astype('float32')
n_features = user.max() + 1 + movie.max() + 1

# Formatting dataset
loc = np.zeros((len(data), 2), dtype='int32')
loc[:, 0] = user
loc[:, 1] = movie + user.max()
val = np.ones((len(data), 2), dtype='float32')

# Train test split
tloc, vloc, tval, vval, ty, vy = train_test_split(loc, val, rating,
                                                  random_state=42)
total_nobs = len(tloc)
