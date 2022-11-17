# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 19:49:25 2022

@author: jules
"""
### SURTOUT NE PAS LANCER SANS LES BONNES CONFIGURATIONS
### Il faut installer sklearn version 2.3 (et pas la derniere version)
import numpy as np
import lazypredict
from lazypredict.Supervised import LazyClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# affichage graphique sur la console
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# chargement des donnees
X = np.load("MNIST_X_28x28.npy")
y = np.load("MNIST_y.npy")

# normalisation
X /= 255.0

# separation des jeux de donnees
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# vectorisation des jeux de donnees
x_train = x_train.reshape(len(x_train),784)
x_test = x_test.reshape(len(x_test),784)

print(x_train.shape)

# Entrainement massif
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(x_train, x_test, y_train, y_test)
models