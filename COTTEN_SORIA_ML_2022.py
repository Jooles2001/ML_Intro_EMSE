#%%
# import modules

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score,adjusted_rand_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import CategoricalNB


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 

from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import v_measure_score



#%%
# on a importé les données dans le même fichier
# load the data

X = np.load("MNIST_X_28x28.npy")
y = np.load("MNIST_y.npy")


#%%
# #plot a sample picture
for nb_sample in range(60):
    plt.subplot(6,10,nb_sample+1)
    plt.imshow(X[nb_sample], cmap=plt.get_cmap('gray'))
    # img_title='Classe ' + str(y[nb_sample])
    # plt.title(img_title)
    plt.axis('off')
plt.show()
plt.clf()


#%%

# data normalization
X /= 255.0

plt.hist(y, bins=np.arange(11) - 0.5, align='mid', rwidth = 0.5, color='purple', edgecolor='black')
plt.xticks(range(10))
plt.show()

#%%
# vectorization des images et division des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
X_train = X_train.reshape(len(X_train), -1)
print(X_train.shape)
X_test = X_test.reshape(len(X_test), -1)

# histogrammes

# plt.hist(y_train, bins=np.arange(11) - 0.5, align='mid', rwidth = 0.5, color='purple', edgecolor='black')
# plt.xticks(range(10))
# plt.show()
# plt.hist(y_test,bins=10, align='right', color='yellow', edgecolor='black')
# plt.show()

#%%

# Principal Component Analysis

pca = PCA(n_components = 784)
pca.fit(X_train)

X_train_PCA = pca.fit_transform(X_train)


## afficher la variance de chaque component

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

plt.figure()
plt.plot(range(pca.n_components),pca.explained_variance_ratio_)
plt.show()
plt.clf()

plt.figure()
PCA_sum = [0]*784
indice = 0
flag = True
for i in range(784):
    PCA_sum[i] += PCA_sum[i-1] + pca.explained_variance_ratio_[i]
    if ((PCA_sum[i] >= 0.95) and flag):
        indice = i
        flag = False

plt.plot(range(784),PCA_sum, color='b')
plt.plot([indice, indice],[0, 1],color='g')
plt.plot(range(784),[0.95]*784,color='r')
plt.show()

print(indice)


#%%
pca = PCA(n_components = 196)
pca.fit(X_train)

X_train_PCA = pca.fit_transform(X_train)
X_test_PCA = pca.fit_transform(X_test)
X_train_PCA = X_train_PCA.reshape(len(X_train_PCA),14,14)

for nb_sample in range(12):
    plt.subplot(3,4,nb_sample+1)
    plt.imshow(X_train_PCA[nb_sample], cmap=plt.get_cmap('gray'))
    plt.axis('off')
    # img_title='Classe ' + str(y[nb_sample])
    # plt.title(img_title)
plt.show()
plt.clf()


#%%

### TOUTES LES CELLULES DU DESSUS DOIVENT ETRE EXECUTEES

#%%
# Unsupervised Learning
print("\n--- Unsupervised Learning ---")
print("--- KMeans Clustering ---")
kmeans = KMeans(n_clusters=10, init='k-means++',n_init=10)
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_test)

# kmeans.labels_

# cf_matrix = confusion_matrix(y_train, kmeans.labels_)

# disp = ConfusionMatrixDisplay(cf_matrix, display_labels=range(10))

# disp.plot()
# plt.show()

print("\nAvant PCA")
training_score = rand_score(y_train,kmeans.labels_)
training_adjusted_score = adjusted_rand_score(y_train,kmeans.labels_)
testing_score = rand_score(y_test,y_kmeans)
testing_adjusted_score = adjusted_rand_score(y_test,y_kmeans)
print("training_score: ", training_score)
print("adjusted_training_score: ", training_adjusted_score)
print("testing_score: ", testing_score)
print("adjusted_testing_score: ", testing_adjusted_score)



#%%
# On évalue sur tous les autres critères
# from https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2020-10-26-01-K-Means-Clustering-for-Imagery-Analysis.ipynb#scrollTo=xuNGWp6m4UM0
# expanded with other metrics that were important to us
def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    # Loop through the clusters
    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
        
    return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

cluster_labels = infer_cluster_labels(kmeans, y_train)
X_clusters = kmeans.predict(X_train)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print(predicted_labels[:20])
print(y_train[:20])


def calc_metrics(estimator, data, labels):
    print('Number of Clusters: {}'.format(estimator.n_clusters))
    # Inertia
    inertia = estimator.inertia_
    print("Inertia: {}".format(inertia))
    # Homogeneity Score
    homogeneity = homogeneity_score(labels, estimator.labels_)
    print("Homogeneity score: {}".format(homogeneity))
    # Completeness Score
    completeness = completeness_score(labels, estimator.labels_)
    print("Completeness score: {}".format(completeness))
    # V_Measure Score
    v_measure = v_measure_score(labels, estimator.labels_)
    print("V_Measure score: {}".format(v_measure))
    return inertia, homogeneity, completeness, v_measure


clusters = [2, 5, 10, 15, 20, 40, 60, 120, 240]
iner_list = []
homo_list = []
comp_list = []
vmes_list = []
acc_list = []

for n_clusters in clusters:
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(X_train)
    
    inertia, homo, compl, vmes = calc_metrics(estimator, X_train, y_train)
    iner_list.append(inertia)
    homo_list.append(homo)
    comp_list.append(compl)
    vmes_list.append(vmes)
    
    # Determine predicted labels
    cluster_labels = infer_cluster_labels(estimator, y_train)
    prediction = infer_data_labels(estimator.labels_, cluster_labels)
    
    acc = accuracy_score(y_train, prediction)
    acc_list.append(acc)
    print('Accuracy: {}\n'.format(acc))

#%%
#affichage
fig, ax = plt.subplots(1, 2, figsize=(16, 10))
ax[0].plot(clusters, iner_list, label='inertia', marker='o')
ax[1].plot(clusters, homo_list, label='homogeneity', marker='o')
ax[1].plot(clusters, acc_list, label='accuracy', marker='^')
ax[1].plot(clusters, comp_list, label='completeness', marker='+')
ax[1].plot(clusters, vmes_list, label='V_measure', marker='*')
ax[0].legend(loc='best')
ax[1].legend(loc='best')
ax[0].grid('on')
ax[1].grid('on')
ax[0].set_title('Inertia of each clusters')
ax[1].set_title('Homogeneity and other metrics of each clusters')
plt.show()

#%%




#%%
# Reduction de Dimension puis KMeans
print("\nKMeans avec K=10 et n_components=2")
pca2 = PCA(n_components=2)
X_train_PCA2 = pca2.fit_transform(X_train)
X_test_PCA2 = pca2.fit_transform(X_test)

kmeans = KMeans(n_clusters=10, init='k-means++',n_init=10)
kmeans.fit(X_train_PCA2)
y_kmeans = kmeans.predict(X_test_PCA2)

training_score = rand_score(y_train,kmeans.labels_)
training_adjusted_score = adjusted_rand_score(y_train,kmeans.labels_)
testing_score = rand_score(y_test,y_kmeans)
testing_adjusted_score = adjusted_rand_score(y_test,y_kmeans)
print("training_score: ", training_score)
print("adjusted_training_score: ", training_adjusted_score)
print("testing_score: ", testing_score)
print("adjusted_testing_score: ", testing_adjusted_score)


#%%
# Matrice de Confusion pour le KMeans
kmeans = KMeans(n_clusters=10, init='k-means++')
kmeans.fit(X_train)
y_kmeans = kmeans.predict(X_test)

training_score = rand_score(y_train,kmeans.labels_)
training_adjusted_score = adjusted_rand_score(y_train,kmeans.labels_)
testing_score = rand_score(y_test,y_kmeans)
testing_adjusted_score = adjusted_rand_score(y_test,y_kmeans)
print("training_score: ", training_score)
print("adjusted_training_score: ", training_adjusted_score)
print("testing_score: ", testing_score)
print("adjusted_testing_score: ", testing_adjusted_score)

cluster_labels = infer_cluster_labels(kmeans, y_train)
prediction = infer_data_labels(kmeans.labels_, cluster_labels)

cm = confusion_matrix(y_train, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.show()


#%%
# New method for clustering
print("Avec EM Gaussian Mixture et PCA")
# PCA
pca = PCA(n_components = 190)
pca.fit(X_train)
X_train_PCA = pca.fit_transform(X_train)
# X_test_PCA = pca.fit_transform(X_test)
# EM Gaussian Mixture
EMGauss = GaussianMixture(n_components=10, random_state=42)
# EMGauss.fit(X_train_PCA)
y_EM_train = EMGauss.predict(X_train_PCA)
# y_EM_test = EMGauss.predict(X_test_PCA)


training_score = rand_score(y_train,y_EM_train)
training_adjusted_score = adjusted_rand_score(y_train,y_EM_train)
# testing_score = rand_score(y_test,y_EM_test)
# testing_adjusted_score = adjusted_rand_score(y_test,y_EM_test)


print("training_score: ", training_score)
print("adjusted_training_score: ", training_adjusted_score)
# print("testing_score: ", testing_score)
# print("adjusted_testing_score: ", testing_adjusted_score)

#%%
# Fonctions d'affichages


def kmeans_graph(X_train):


    kmeans = KMeans(10, init='random').fit(X_train) #k-means++
    label = kmeans.predict(X_train)

    pca = PCA(2)
    X_train = pca.fit_transform(X_train)
    
    centers = kmeans.cluster_centers_
    for i in range(10):
        plt.scatter(X_train[label == i, 0], X_train[label == i, 1], label = i, s=5)
    plt.scatter(centers[:,0] , centers[:,1] , s = 50, color = 'w')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(1, 10, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(10, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()

kmeans_graph(X_train)
    
def EM_graph(X_train):
    pca = PCA(2)
    X_train = pca.fit_transform(X_train)
    em = GaussianMixture(n_components = 10, covariance_type = 'full', n_init=5).fit(X_train)
    label = em.predict(X_train)
    #centers = em.cluster_centers_
    for i in range(10):
        plt.scatter(X_train[label == i, 0], X_train[label == i, 1], label = i, s=5)
    #plt.scatter(centers[:,0] , centers[:,1] , s = 50, color = 'w')
    plt.legend()
    plt.show()

EM_graph(X_train)
# Courtoisie de Leo-Paul Hauet



#%%
# Unsupervised Deep Learning
# Autoencoders

#########
## Cette partie se fait dans un code annexe car il est très long et est en dehors du cadre du TP
#########



#%%
# Supervised Learning

# Decision Trees

pca = PCA(n_components=153)
Xn = pca.fit_transform(X_train)
Xn_train, Xn_test, Yn_train, Yn_test = train_test_split(Xn, y_train, test_size=0.2)

print("--- Decision Tree ---")
clf = tree.DecisionTreeClassifier(max_leaf_nodes=1000)
clf = clf.fit(Xn_train, Yn_train)
print("characteristics: \ndepth: " + str(clf.get_depth()) + "\nn_leafs: " + str(clf.get_n_leaves()))
print("accuracy on training set: " + str(clf.score(Xn_train, Yn_train)))
print("accuracy on testing set: " + str(clf.score(Xn_test, Yn_test)))

print("--- Decision Tree model---")
clf2 = tree.DecisionTreeClassifier(max_leaf_nodes=1000)
clf2 = clf2.fit(X_train, y_train)
print("characteristics: \ndepth: " + str(clf2.get_depth()) + "\nn_leafs: " + str(clf2.get_n_leaves()))
print("accuracy on training set: " + str(clf2.score(X_train, y_train)))
print("accuracy on testing set: " + str(clf2.score(X_test, y_test)))
plt.show()

print("--- Decision Tree model simple---")
clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(X_train, y_train)
print("characteristics: \ndepth: " + str(clf3.get_depth()) + "\nn_leafs: " + str(clf3.get_n_leaves()))
print("accuracy on training set: " + str(clf3.score(X_train, y_train)))
print("accuracy on testing set: " + str(clf3.score(X_test, y_test)))
plt.show()

#%%
# Support Vector Machines
# SVM hard margin
print("\n\nSupervised Learning")



# y_svm = clf.predict(X_test)
print("\n--- SVM ---")
print("\nLinear Hard-Margin")
clf = svm.SVC(kernel="linear", C=1)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("\nTraining Dataset:")
print(str(clf.score(X_train,y_train)))
print("\nTesting Dataset:")
print(str(clf.score(X_test,y_test)))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%%
# SVM soft margin
print("\nLinear Soft-Margin")
clf = svm.SVC(kernel="linear", C = 0.05)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("\nTraining Dataset:")
print(str(clf.score(X_train,y_train)))
print("\nTesting Dataset:")
print(str(clf.score(X_test,y_test)))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%%
# SVM RBF
print("\nRadial Basis Function")
clf = svm.SVC(kernel="rbf")
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("\nTraining Dataset:")
print(str(clf.score(X_train,y_train)))
print("\nTesting Dataset:")
print(str(clf.score(X_test,y_test)))
# print(rand_score(y_test,y_svm))
# print(adjusted_rand_score(y_test,y_svm))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%%

# Logistic Regression

print("--- Logistic regression ---")
clf = LogisticRegression(random_state=0, solver='sag', multi_class='multinomial').fit(X_train, y_train)
print("accuracy on training set: " + str(clf.score(X_train, y_train)))
print("accuracy on testing set: " + str(clf.score(X_test, y_test)))


#%%
# Naive Bayes Classifier
print("\nNaive Bayes Classifier - Gaussian")
modelNB = GaussianNB()
modelNB.fit(X_train, y_train)
y_pred = modelNB.predict(X_test)
print("\nTraining Dataset:")
print(str(modelNB.score(X_train,y_train)))
print("\nTesting Dataset:")
print(str(modelNB.score(X_test,y_test)))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


#%%
# K-Nearest Neighbors
print("\nK-Nearest Neighbors")
for K in [2,5,8,10,16]:
    modelKNN = KNeighborsClassifier(n_neighbors=K)
    modelKNN.fit(X_train, y_train)
    y_pred = modelKNN.predict(X_test)
    print("\nK = {0}".format(K))
    print("Training Dataset:")
    print(str(modelKNN.score(X_train,y_train)))
    print("Testing Dataset:")
    print(str(modelKNN.score(X_test,y_test)))

cm = confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%%
# Deep Learning

### CELLULE IMPORTANTE

print("\n\n--- Deep Learning ---")


y_train = keras.utils.to_categorical(y_train) # transforme en hot-vector
y_test = keras.utils.to_categorical(y_test) # A NE PAS ECRIRE SI ON UTILISE SparseCategoricalCrossentropy(from_logits=False)
# on utilise keras.models et keras.layers 

#%%

# MLP

print("MLP tensorflow")
#MODELE 2 HIDDEN LAYER
dropout_value = 0.40
input_layer = keras.Input(shape=(784,))
x = layers.Dropout(0.2)(input_layer)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(dropout_value)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(dropout_value)(x)
output_layer = layers.Dense(10, activation="softmax")(x)
model = keras.Model(input_layer, output_layer, name="mnist_model")
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

#MODELE 1 HIDDEN LAYER
# input_layer = keras.Input(shape=(784,))
# x = layers.Dropout(0.25)(input_layer)
# x = layers.Dense(512, activation="relu")(x)
# x = layers.Dropout(0.5)(x)
# output_layer = layers.Dense(10, activation="softmax")(x)
# model = keras.Model(input_layer, output_layer, name="mnist_model")
# model.summary()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )


nb_epochs = 100
history = model.fit(X_train, y_train, batch_size=64, epochs=nb_epochs, validation_split=0.2)
test_scores_train = model.evaluate(X_train, y_train, verbose=2)
print("Train loss:", test_scores_train[0])
print("Train accuracy:", test_scores_train[1])
test_scores_test = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_scores_test[0])
print("Test accuracy:", test_scores_test[1])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.plot(range(0, nb_epochs), history.history['accuracy'])
plt.plot(range(0, nb_epochs), history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='best')
plt.show()


#%%
# MLP QUI OVERFIT

print("\n--- Multi-Layer Perceptron ---")
model = models.Sequential()
model.add(keras.Input(784,))
model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.Dropout(0.2))
model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
# model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
history = model.fit(X_train,y_train,validation_split=0.2,epochs=40)
history_dict = history.history

# ON A ESSAYE DE GENERER DE L'OVERFITTING ICI

results = model.evaluate(X_test,y_test)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.clf()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.clf()


#%%
# ETAPE TRES IMPORTANTE AVANT LE LANCEMENT DES MODELES CONVOLUTIONNEL
# CNN
print("\n--- Convolutional Neural Network ---")
print("Recreation d'image a partir de vecteurs")
print(X_train.shape)
X_train = X_train.reshape(len(X_train),28,28,1)
print(X_train.shape)
X_test = X_test.reshape(len(X_test),28,28,1)

#%%
# On va commencer par un modèle simple
# 16 filtres de taille 2x2 pour faire simple 
# padding = 'valid' car on ne veut pas influencer les bordures
# 12 epochs seulement d'abord

model = Sequential()
model.add(Conv2D(16, (2,2), padding='valid',activation='relu',input_shape=X_train.shape[1:]))
model.add(Flatten()) 
model.add(Dense(10, activation = 'softmax'))
print(model.summary())


model.compile(loss = keras.losses.CategoricalCrossentropy(), 
   optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(
   X_train, y_train, 
   batch_size = 128, 
   epochs = 12, 
   verbose = 1, 
   validation_split = 0.2
)

history_dict = history.history


results = model.evaluate(X_test,y_test)

print(history.history.keys())

fig, ax = plt.subplots(1, 2, figsize=(16, 10))
#  "Accuracy"
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'validation'], loc='upper left')

# "Loss"
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'validation'], loc='upper left')


#affichage
plt.show()
plt.clf()


#%%
# Sequential API
### MODELE HAUTE PERFORMANCE
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same',activation='relu',input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2,2),padding='valid'))
model.add(Dropout(0.1))
model.add(Conv2D(256, (2,2), padding='same',activation='relu'))
model.add(MaxPooling2D((2,2),padding='valid'))
model.add(Dropout(0.3))
model.add(Conv2D(64, (2,2), padding='same',activation='relu'))
model.add(MaxPooling2D((2,2),padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation = 'relu',kernel_regularizer=regularizers.l2(0.0003))) 
model.add(Dropout(0.3))
model.add(Dense(192, activation = 'relu',kernel_regularizer=regularizers.l2(0.0003))) 
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu',kernel_regularizer=regularizers.l2(0.0003))) 
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu',kernel_regularizer=regularizers.l2(0.0003))) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation = 'softmax'))
print(model.summary())

model.compile(loss = keras.losses.CategoricalCrossentropy(), 
   optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(
   X_train, y_train, 
   batch_size = 128, 
   epochs = 25, 
   verbose = 1, 
   validation_split = 0.2
)


history_dict = history.history


results = model.evaluate(X_test,y_test)

fig, ax = plt.subplots(1, 2, figsize=(16, 10))
#  "Accuracy"
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'validation'], loc='upper left')

# "Loss"
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'validation'], loc='upper left')


#affichage
plt.show()
plt.clf()

# %%
# END OF CODE