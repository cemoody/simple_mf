# To run this notebook, you'll need to download the urban dictionary dataset.
# You can do that by registering for Kaggle and downloading
# the dataset from here:
# https://www.kaggle.com/therohk/urban-dictionary-words-dataset
# You'll also need to install Spacy & run:
# python -m spacy download en_core_web_sm

import spacy
import os.path
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


def count_skipgrams(docs, unigrams):
    skipgrams = {}
    for doc in tqdm.tqdm(docs):
        tokens = nlp.tokenizer(doc)
        lemmas = [token.lemma_.translate(translator).lower()
                  for token in tokens]
        lemmas = [l for l in lemmas if l in unigrams and len(l) > 2]
        for i, token1 in enumerate(lemmas):
            for j, token2 in enumerate(lemmas):
                # Don't double-count
                if j >= i:
                    break
                key = (token1, token2)
                skipgrams[key] = skipgrams.get(key, 0) + 1
    return skipgrams


def to_dataframe(skipgrams, unigrams):
    t0 = [t[0] for t in skipgrams.keys()]
    t1 = [t[1] for t in skipgrams.keys()]
    cnt = [v for v in skipgrams.values()]
    df = pd.DataFrame(dict(token0=t0, token1=t1, counts=cnt))
    t0 = [t for t in unigrams.keys()]
    cnt = np.array([v for v in unigrams.values()])
    prob0 = cnt * 1.0 / cnt.sum()
    ug0 = pd.DataFrame(dict(token0=t0, prob0=prob0))
    ug1 = pd.DataFrame(dict(token1=t0, prob1=prob0))
    df = df.merge(ug0, on='token0')
    df = df.merge(ug1, on='token1')
    df['prob_sg'] = df['counts'] * 1.0 / df['counts'].sum()
    df['pmi'] = np.log(df.prob_sg / df.prob0 / df.prob1) * 1e6
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
    full['pmi'] = full.pmi_x + full.pmi_y
    del full['counts_x']
    del full['counts_y']
    del full['pmi_x']
    del full['pmi_y']
    return full


def to_codes(full):
    cats = np.concatenate((full.token0.unique(), full.token1.unique()))
    cats = np.unique(cats)
    full['token0_code'] = pd.Categorical(full.token0, categories=cats).codes
    full['token1_code'] = pd.Categorical(full.token1, categories=cats).codes
    code_code_count = full[['token0_code', 'token1_code', 'counts', 'pmi']]
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
    idx &= full.token0 != full.token1
    limited = full[idx]
    limited = limited[limited['counts'] > 0]
    return limited


if __name__ == '__main__':
    fn = 'data/urbandict-word-def.csv'
    docs = textify(fn)
    unigrams = pd.read_pickle('data/unigrams.pd')
    unigrams = unigrams.set_index('token').to_dict()['cnt']
    limited = {k: v for (k, v) in unigrams.items() if v > 100}
    sg = count_skipgrams(docs, limited)
    full = to_dataframe(sg, unigrams)
    full.to_pickle(f'data/skipgrams_p0.pd')
    coded, c2t, t2c = to_codes(full)
    np.savez('data/skipgram_full.npz', coded=coded,
             c2t=c2t, t2c=t2c)
