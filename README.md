Simple and Flexible Deep Recommenders in PyTorch
==============================

![profile](https://i.imgur.com/MWSyBfS.png)

Simple but flexible Deep Recommenders in PyTorch.

Plese review the deck to see the accompanying written & visual content. ![deck](https://i.imgur.com/VqmfR4H.png)


[View the deck here](https://docs.google.com/presentation/d/1gv7osHoSX8CHf0uzKSqOlxmmAvPPdmstL0nrZHWiHQM/edit#slide=id.p)

Check out the notebooks within to step through variations of matrix factorization models. Here's what we'll cover:

0. [Step 0] Introduction to autograd & deep learning using PyTorch, the Ignite library, and recommendation engines.
1. [Step 1] Build a simple matrix-factorization model in PyTorch. These models are a fundamental core to Netflix's, Pandora's, Stitch Fix's and Amazon's recommendations engines.
2. [Step 2] We'll expand on that model to include biases for extra predictive power
4. [Step 3] Add in "side" features, especially useful in coldstart cases
5. [Step 4] Model temporal effects which can track seasonal and periodic changes.
3. [Step 5] We'll take detour and see how word2vec is mathematically identical to recommendation engines
6. [Step 6] Upgrade the core of matrix factorization to Factorization Machines, which enables a huge number of interactions while keeping computation under control.
8. [Step 7] We'll wrap up with Bayesian Deep Learning applied to rec engines. This Variational Matrix Factorization is a great way to dip your toes into explore & exploit problems.
8. [Step 8] We'll build a real-time recommender using Transformers to read in an input ratings stream and generate recommendations.

# To get started.
If at all possible, please check out and pre-install the environment.

```
git clone https://github.com/cemoody/simple_mf.git
cd simple_mf
```

## 1. Create environment.
Create the environment by following the steps below. If you choose to use your own environment, you'll need access to have the Python packages in `requirements.txt` installed.


Make sure you you have pytorch installed; if not, follow the instructions [here](https://pytorch.org/get-started/locally/)

```
pip install pytorch-lightning
```

Follow the directions the above command spits out.

## 2. Setup W & B account

You'll be creating using a (free) weights & biases account to track model metrics and performance over time. TO kickstart that process:

```
pip install wandb
wandb login
```

Setup your W&B account, then go to the W&B authorization page: https://app.wandb.ai/authorize and copy the auth code into your terminal when prompted by `wandb login`

## 3. Download and preprocess data:
This will download and preprocess the MovieLens 1M dataset. We'll use this canonical dataset to test drive our code.

```
  # required. will download the movielens 1M dataset.
  python src/download.py

  # optional! this is a bigger dataset we'll use for more
  # advanced models.
  python src/download_ml20.py

  # optional too. This is used for word2vec notebook.
  python src/skipgram.py
```

## 4. Does it work?

Open up and execute every line within the `01 MF model.ipynb` notebook. If it works, you're golden.
