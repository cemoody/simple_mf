Simple and Flexible Deep Recommenders in PyTorch
==============================

![profile](https://i.imgur.com/MWSyBfS.png)

Simple but flexible Deep Recommenders in PyTorch. This hands-on tutorial will teach you:

0. [Step 0]() Introduction to autograd & deep learning using PyTorch, the Ignite library, and recommendation engines.
1. [Step 1](https://github.com/cemoody/simple_mf/blob/master/notebooks/01%20Training%20a%20simple%20MF%20model.ipynb) Build a simple matrix-factorization model in PyTorch. These models are a fundamental core to Netflix's, Pandora's, Stitch Fix's and Amazon's recommendations engines.
2. [Step 2](https://github.com/cemoody/simple_mf/blob/master/notebooks/02%20Simple%20MF%20Model%20with%20Biases.ipynb) We'll expand on that model to include biases for extra predictive power
3. [Step 3](https://github.com/cemoody/simple_mf/blob/master/notebooks/03%20Simple%20MF%20Model%20for%20use%20in%20NLP%20%5BTODO%5D.ipynb) We'll take detour and see how word2vec is mathematically identical to recommendation engines
4. [Step 4](https://github.com/cemoody/simple_mf/blob/master/notebooks/04%20MF%20model%20with%20side-features.ipynb) Add in "side" features, especially useful in coldstart cases
5. [Step 5](https://github.com/cemoody/simple_mf/blob/master/notebooks/05%20MF%20model%20plus%20temporal-features.ipynb) Model temporal effects which can track seasonal and periodic changes.
6. [Step 6](https://github.com/cemoody/simple_mf/blob/master/notebooks/06%20FM%20model.ipynb) Upgrade the core of matrix factorization to Factorization Machines, which enables a huge number of interactions while keeping computation under control.
7. [Step 7](https://github.com/cemoody/simple_mf/blob/master/notebooks/07%20Mixture%20of%20Tastes.ipynb) We'll try out the new "Mixture-of-Tastes" model.
8. [Step 8](https://github.com/cemoody/simple_mf/blob/master/notebooks/08%20Variational%20MF.ipynb) We'll wrap up with Bayesian Deep Learning applied to rec engines. This Variational Matrix Factorization is a great way to dip your toes into explore & exploit problems.

# To get started.
If at all possible, please check out and pre-install the environment.

```
git clone https://github.com/cemoody/simple_mf.git
cd simple_mf
```

## 1. Create environment.
Create the environment by following the steps below. If you choose to use your own environment, you'll need access to have the Python packages in `requirements.txt` installed.
```
make create_environment
```

Follow the directions the above command spits out.

## 2. Activate the environment.
The output of the last command, which depends on if you're using `conda` or not, will tell you how to activate your environment.

**Follow the steps in #1 carefully -- you probably don't need step 2a. or 2b!**

### 2a. With `conda`.

If the above step used conda, you can active it the conda environment by running:

`source activate simple_mf`

### 2b. With `pip`.
If you don't have `conda`, then the output of the `make create_environment` will spit out something like this:
```
Make sure the following lines are in shell startup file
export WORKON_HOME=/Users/chrismoody/.virtualenvs
export PROJECT_HOME=/Users/chrismoody/Devel
source /usr/local/bin/virtualenvwrapper.sh
workon simple_mf
```

Go ahead and activate your environment by running the above commands.



## 3. Download data:
This will download and preprocess the MovieLens 1M dataset. We'll use this canonical dataset to test drive our code.

```
  make download
```

## 4. Run tensorboard in the background.
While we're using PyTorch instead of Tensorflow directly, the logging and visualization library Tensorboard is an amazing asset to track the progress of our models. It's implemented as a small local web server that constructs visualizations from log files, so start by kicking it off in the background:

```
cd notebooks
tensorboard --logdir runs
```

Visit the tensorboard dashbaord by going to [http://localhost:6006](http://localhost:6006)

## 5. Run Jupyter Notebook locally.

This will startup the Jupyter server and open up the available notebooks. Try running a few notebooks ahead of time to verify that your environment is setup and functioning well.

```
jupyter notebook
```

Visit the jupyter notebooks by going [http://localhost:8888/tree](http://localhost:8888/tree)
