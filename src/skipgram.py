# To run this notebook, you'll need to download the urban dictionary dataset.
# You can do that by registering for Kaggle and downloading
# the dataset from here:
# https://www.kaggle.com/therohk/urban-dictionary-words-dataset
# You'll also need to install Spacy & run:
# python -m spacy download en_core_web_sm

from collections import defaultdict
import spacy
import numpy as np
import pandas as pd
import string
import tqdm
import random
from sklearn.externals import joblib

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load('en_core_web_sm')
translator = str.maketrans('', '', string.punctuation)
mem = joblib.Memory('cache')


@mem.cache
def textify(fn):
    docs = []
    with open(fn, 'r') as fh:
        for j, line in enumerate(fh):
            # Skip first fields
            splits = line.split(',')
            word = splits[1]
            definition = ','.join(splits[5:])
            definition = definition.replace('"', '').replace('\n', '')
            docs.append(definition + ' ' + word)
    return docs


def tokenize(docs):
    n_docs = len(docs)
    tokenized = [nlp.tokenizer(doc)
                 for doc in tqdm.tqdm(docs, total=n_docs)]
    return tokenized


def count_skipgrams(docs):
    skipgrams = {}
    for doc in tqdm.tqdm(docs):
        tokens = nlp.tokenizer(doc)
        lemmas = [token.lemma_.translate(translator).lower()
                  for token in tokens if len(token.lemma_) > 2]
        for i, token1 in enumerate(lemmas):
            for j, token2 in enumerate(lemmas):
                # Don't double-count
                if j >= i:
                    break
                key = (token1, token2)
                if key in skipgrams:
                    skipgrams[key] += 1
                # We're stochastically down-sampling rare skipgrams
                elif random.random() < 0.50:
                    skipgrams[key] = 0
    return skipgrams


def to_dataframe(skipgrams):
    t0 = [t[0] for t in skipgrams.keys()]
    t1 = [t[1] for t in skipgrams.keys()]
    cnt = [v for v in skipgrams.values()]
    df = pd.DataFrame(dict(token0=t0, token1=t1, counts=cnt))
    return df


def merge(n=2):
    full = None
    for start in range(n):
        fn = f'data/skipgrams_p{start}.pd'
        df = pd.read_pickle(fn)
        if full is None:
            full = df
        else:
            key = ('token0', 'token1')
            full = (full.merge(df, on=key, how='outer')
                        .fillna(0.0)
                        .reset_index())
    full['counts'] = full.counts_x + full.counts_y
    del full['counts_x']
    del full['counts_y']
    return full


def to_codes(full):
    cats = np.concatenate((full.token0.unique(), full.token1.unique()))
    cats = np.unique(cats)
    full['token0_code'] = pd.Categorical(full.token0, categories=cats).codes
    full['token1_code'] = pd.Categorical(full.token1, categories=cats).codes
    code_code_count = full[['token0_code', 'token1_code', 'counts']]
    code_code_count = code_code_count.values.astype(np.int32)
    code2token = {}
    code2token.update({code: token for (code, token)
                       in zip(full.token0_code, full.token0)})
    code2token.update({code: token for (code, token)
                       in zip(full.token1_code, full.token1)})
    token2code = {token: code for (code, token) in code2token.items()}
    return code_code_count, code2token, token2code


def high_counts_only(full, n=10000):
    token_counts = full.groupby('token0')['counts'].sum().sort_values()
    top_words = set(token_counts[-n:].index)
    idx = full.token0.isin(top_words)
    idx &= full.token1.isin(top_words)
    limited = full[idx]
    limited = limited[limited['counts'] > 0]
    return limited


if __name__ == '__main__':
    fn = 'data/urbandict-word-def.csv'
    # docs = textify(fn)
    # tokenized = tokenize(docs)
    # sg = tokenize(docs)
    # for start in range(2):
    #     sg = count_skipgrams(docs[1::2])
    #     df = to_dataframe(sg)
    #     df.to_pickle(f'data/skipgrams_p{start}.pd')
    full = merge()
    sub = high_counts_only(full)
    coded, c2t, t2c = to_codes(sub)
    np.savez('data/skipgram_full.npz', coded=coded,
             c2t=c2t, t2c=t2c)
