#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
from scipy.misc import imread
import tensorflow as tf
from os import listdir
from sklearn.model_selection import RepeatedKFold
from skimage.transform import resize
from keras.utils import Sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.initializers import VarianceScaling
from sklearn.metrics import accuracy_score as accuracy
import keras.backend as K
from keras.callbacks import Callback

import os
print(os.listdir("../input/hpaic"))


# In[2]:


sns.set()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[3]:


train_labels = pd.read_csv("../input/hpaic/train.csv")
train_labels.head()


# In[29]:


# find number of samples
train_labels.shape[0]


# In[4]:


train_path = "../input/hpaic/train/"
test_path = "../input/hpaic/test/"

submission = pd.read_csv("../input/hpaic/sample_submission.csv")
submission.head()


# In[5]:


# find number of testing samples
test_names = submission.Id.values
print(len(test_names))
print(test_names[0])


# In[6]:


label_names = {
    0: "Nucleoplasm",  
    1: "Nuclear membrane",   
    2: "Nucleoli",   
    3: "Nucleoli fibrillar center",   
    4: "Nuclear speckles",
    5: "Nuclear bodies",   
    6: "Endoplasmic reticulum",   
    7: "Golgi apparatus",   
    8: "Peroxisomes",   
    9: "Endosomes",   
    10: "Lysosomes",   
    11: "Intermediate filaments",   
    12: "Actin filaments",   
    13: "Focal adhesion sites",   
    14: "Microtubules",   
    15: "Microtubule ends",   
    16: "Cytokinetic bridge",   
    17: "Mitotic spindle",   
    18: "Microtubule organizing center",   
    19: "Centrosome",   
    20: "Lipid droplets",   
    21: "Plasma membrane",   
    22: "Cell junctions",   
    23: "Mitochondria",   
    24: "Aggresome",   
    25: "Cytosol",   
    26: "Cytoplasmic bodies",   
    27: "Rods & rings"
}


# In[7]:


# make target values binary
for key in label_names.keys():
    train_labels[label_names[key]] = 0


# In[8]:


def add_target(row):
    row.Target = np.array(row.Target.split()).astype(np.int)
    for x in row.Target:
        label = label_names[int(x)]
        row.loc[label] = 1
    return row

train_labels = train_labels.apply(add_target, axis=1)
train_labels.head()


# In[9]:


test_labels = pd.DataFrame(data=test_names, columns=["Id"])

for col in train_labels.columns.values:
    if col != "Id":
        test_labels[col] = 0

test_labels.head()


# In[10]:


# get insights on data

# which target is how common?

target_counts = train_labels.drop(["Id", "Target"], axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)


# In[30]:


# how many targets are most common?

train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"], axis=1).sum(axis=1)
count_percent = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_percent.index.values, y=count_percent.values, palette="Blues")
plt.xlabel("number of targets per image")
plt.ylabel("% of train data")


# In[31]:


# find corelation

plt.figure(figsize=(15,15))
sns.heatmap(train_labels[train_labels.number_of_targets > 1].drop(["Id", "Target", "number_of_targets"],axis=1).corr(), cmap="RdBu", vmin=-1, vmax=1)


# In[32]:


# how often are special targets together?

def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(["Id", "Target", "number_of_targets"],axis=1).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts


# In[33]:


lyso_endo_counts = find_counts("Lysosomes", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues")
plt.ylabel("counts in train data")


# In[34]:


rod_rings_counts = find_counts("Rods & rings", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=rod_rings_counts.index.values, y=rod_rings_counts.values, palette="Blues")
plt.ylabel("counts in train data")


# In[35]:


peroxi_counts = find_counts("Peroxisomes", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=peroxi_counts.index.values, y=peroxi_counts.values, palette="Blues")
plt.ylabel("counts in train data")


# In[36]:


tubeends_counts = find_counts("Microtubule ends", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=tubeends_counts.index.values, y=tubeends_counts.values, palette="Blues")
plt.ylabel("counts in train data")


# In[37]:


nuclear_speckles_counts = find_counts("Nuclear speckles", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=nuclear_speckles_counts.index.values, y=nuclear_speckles_counts.values, palette="Blues")
plt.xticks(rotation="70")
plt.ylabel("counts in train data");


# In[38]:


# how do the images look?

files = listdir(train_path)
for n in range(10):
    print(files[n])


# In[39]:


# since we have one file for each of red, green, blue, and yellow channels, total number of files divided by 4 should yield the number of target samples

len(files) / 4 == train_labels.shape[0]


# In[40]:


# load images in a batch

def load_image(path, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(path + image_id + "_green" + ".png")
    images[1,:,:] = imread(path + image_id + "_red" + ".png")
    images[2,:,:] = imread(path + image_id + "_blue" + ".png")
    images[3,:,:] = imread(path + image_id + "_yellow" + ".png")
    return images


# In[41]:


def make_image_row(image, ax, title):
    ax[0].imshow(image[0], cmap="Greens")
    ax[1].imshow(image[1], cmap="Reds")
    ax[1].set_title("stained microtubules")
    ax[2].imshow(image[2], cmap="Blues")
    ax[2].set_title("stained nucleus")
    ax[3].imshow(image[3], cmap="YlOrBr")
    ax[3].set_title("stained endoplasmatic reticulum")
    ax[0].set_title(title)
    return ax


# In[46]:


def make_title(file_id):
    file_targets = train_labels.loc[train_labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title


# In[47]:


class image_loader:
    
    def __init__(self, target_names, batch_size, path):
        reverse_lables = dict((v,k) for k,v in label_names.items())
        self.target_names = target_names
        self.target_list = [reverse_lables[key] for key in target_names]
        self.batch_shape = (batch_size, 4, 512, 512)
        self.path = path
    
    def find_matches(self):
        train_labels["check_col"] = train_labels.Target.apply(lambda l: self.check_subset(l))
        self.images_identifier = train_labels[train_labels.check_col==1].Id.values
        train_labels.drop("check_col", axis=1, inplace=True)
    
    def check_subset(self, targets):
        return np.where(set(targets).issubset(set(self.target_list)), 1, 0)
    
    def get_loader(self):
        files = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for image_id in self.images_identifier:
            images[idx,:,:,:] = load_image(self.path, image_id)
            files.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield files, images
                files = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield files, images


# In[48]:


imageloader = image_loader(["Lysosomes", "Endosomes"], 5, train_path)
imageloader.find_matches()
iterator = imageloader.get_loader()


# In[49]:


file_ids, images = next(iterator)

fig, ax = plt.subplots(len(file_ids), 4, figsize=(20, 5 * len(file_ids)))
if ax.shape == (4,):
    ax = ax.reshape(1, -1)
for n in range(len(file_ids)):
    make_image_row(images[n], ax[n], make_title(file_ids[n]))


# In[11]:


# begin model

train_files = listdir(train_path)
test_files = listdir(test_path)

print(len(test_files) / len(train_files) * 100)


# In[12]:


# k-fold cross validation

splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
partitions = []

for train_idx, test_idx in splitter.split(train_labels.index.values):
    partition = {}
    partition["train"] = train_labels.Id.values[train_idx]
    partition["test"] = train_labels.Id.values[test_idx]
    partitions.append(partition)
    print("train partition:", len(train_idx), "| test partition:", len(test_idx))


# In[50]:


partitions[0]["train"][0:5]


# In[51]:


# parameters for the model

class model_parameters:
    
    def __init__(self, path, num_classes=28, image_rows=512, image_cols=512, batch_size=200, 
                 n_channels=1, row_scale_factor=4, col_scale_factor=4, shuffle=False,
                 n_epochs=1):
        self.path = path
        self.num_classes = num_classes
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.row_scale_factor = row_scale_factor
        self.col_scale_factor = col_scale_factor
        self.scaled_row_dim = np.int(self.image_rows / self.row_scale_factor)
        self.scaled_col_dim = np.int(self.image_cols / self.col_scale_factor)
        self.n_epochs = n_epochs


# In[52]:


params = model_parameters (train_path)


# In[53]:


# preprocess the image to scale it down

class image_preprocessor:
    
    def __init__(self, params):
        self.params = params
        self.path = self.params.path
        self.scaled_row_dim = self.params.scaled_row_dim
        self.scaled_col_dim = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
    
    def preprocess(self, image):
        image = self.resize(image)
        image = self.reshape(image)
        image = self.normalize(image)
        return image
    
    def resize(self, image):
        image = resize(image, (self.scaled_row_dim, self.scaled_col_dim))
        return image
    
    def reshape(self, image):
        image = np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
        return image
    
    def normalize(self, image):
        image /= 255 
        return image
    
    def load_image(self, image_id):
        image = np.zeros(shape=(512,512,4))
        image[:,:,0] = imread(self.path + image_id + "_green" + ".png")
        image[:,:,1] = imread(self.path + image_id + "_blue" + ".png")
        image[:,:,2] = imread(self.path + image_id + "_red" + ".png")
        image[:,:,3] = imread(self.path + image_id + "_yellow" + ".png")
        return image[:,:,0:self.params.n_channels]


# In[54]:


ipp = image_preprocessor (params)


# In[56]:


example = images[0,0]
preprocessed = ipp.preprocess(example)
print(example.shape)
print(preprocessed.shape)

fig, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].imshow(example, cmap="Greens")
ax[1].imshow(preprocessed.reshape(params.scaled_row_dim, params.scaled_col_dim), cmap="Greens")


# In[57]:


class data_generator(Sequence):
    
    def __init__(self, list_IDs, labels, params, ipp):
        self.current_epoch = 0
        self.params = params
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = (self.params.scaled_row_dim, self.params.scaled_col_dim)
        self.batch_size = self.params.batch_size
        self.n_channels = self.params.n_channels
        self.num_classes = self.params.num_classes
        self.shuffle = self.params.shuffle
        self.ipp = ipp
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes, random_state=self.current_epoch)
            self.current_epoch += 1
    
    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id==identifier].drop(["Id", "Target", "number_of_targets"], axis=1).values
            
    def __data_generation(self, list_IDs_temp):
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        for i, identifier in enumerate(list_IDs_temp):
            image = self.ipp.load_image(identifier)
            image = self.ipp.preprocess(image)
            x[i] = image
            y[i] = self.get_targets_per_image(identifier)
        return x, y
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        x, y = self.__data_generation(list_IDs_temp)
        return x, y


# In[16]:


class predict_generator:
    
    def __init__(self, predict_Ids, ipp, path):
        self.ipp = ipp
        self.ipp.path = path
        self.identifiers = predict_Ids
    
    def predict(self, model):
        y = np.empty(shape=(len(self.identifiers), self.ipp.params.num_classes))
        for n in range(len(self.identifiers)):
            image = self.ipp.load_image(self.identifiers[n])
            image = self.ipp.preprocess(image)
            image = image.reshape((1, *image.shape))
            y[n] = model.predict(image)
        return y


# In[17]:


class CNN:
    
    def __init__(self, params):
        self.params = params
        self.num_classes = self.params.num_classes
        self.img_rows = self.params.scaled_row_dim
        self.img_cols = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
        self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        self.evaluation_metrics = ['accuracy']
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                             kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
    
    def compile_model(self):
        self.model.compile(loss=binary_crossentropy, optimizer=Adadelta(), metrics=self.evaluation_metrics)
    
    def set_generators(self, train_gen, test_gen):
        self.train_gen = train_gen
        self.test_gen = test_gen
    
    def learn(self):
        return self.model.fit_generator(generator=self.train_gen, validation_data=self.test_gen,
                                        epochs=self.params.n_epochs,  use_multiprocessing=True, workers=8)
    
    def score(self):
        return self.model.evaluate_generator(generator=self.test_gen, use_multiprocessing=True, 
                                             workers=8)
    
    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y


# In[58]:


partition = partitions[0]

print("Number of samples in train:", len(partition["train"]))
print("Number of samples in test:", len(partition["test"]))


# In[59]:


train_gen = data_generator(partition['train'], train_labels, params, ipp)
test_gen = data_generator(partition['test'], train_labels, params, ipp)
predict_gen = predict_generator(partition['test'], ipp, train_path)


# In[60]:


tipp = image_preprocessor(params)
submission_predict_gen = predict_generator(test_names, tipp, test_path)


# In[61]:


model = CNN(params)
model.build_model()
model.compile_model()
model.set_generators(train_gen, test_gen)
history = model.learn()

prob_predictions = model.predict(predict_gen)
CNN_prob_predictions = pd.DataFrame(index=partition['test'], data=prob_predictions, columns=train_labels.drop(["Target", "number_of_targets", "Id"], axis=1).columns)
# CNN_prob_predictions.to_csv("CNN_predictions.csv")
CNN_losses = pd.DataFrame(history.history["loss"], columns=["train_loss"])
CNN_losses["val_loss"] = history.history["val_loss"]
# CNN_losses.to_csv("CNN_losses.csv")

submission_prob_predictions = model.predict(submission_predict_gen)
CNN_labels = test_labels.copy()
CNN_labels.loc[:, test_labels.drop(["Id", "Target"], axis=1).columns.values] = submission_prob_predictions
# CNN_labels.to_csv("CNN_submission_prob.csv")


# In[62]:


validation_labels = train_labels.loc[train_labels.Id.isin(partition["test"])].copy()
validation_labels.shape


# In[63]:


CNN_prob_predictions.shape


# In[64]:


y_true = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).values
y_test = np.where(CNN_prob_predictions.values > 0.5, 1, 0)
accuracy(y_true.flatten(), y_test.flatten())


# In[65]:


y_test[0]


# In[66]:


y_true[0]


# In[67]:


prob_predictions = CNN_prob_predictions.values
hot_values = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).values.flatten()
one_hot = (hot_values.sum()) / hot_values.shape[0] * 100
zero_hot = (hot_values.shape[0] - hot_values.sum()) / hot_values.shape[0] * 100

fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(prob_predictions.flatten() * 100, color="DodgerBlue", ax=ax[0])
ax[0].set_xlabel("Probability in %")
ax[0].set_ylabel("Density")
ax[0].set_title("Predicted probabilities")
sns.barplot(x=["label = 0", "label = 1"], y=[zero_hot, one_hot], ax=ax[1])
ax[1].set_ylim([0,100])
ax[1].set_title("True target label count")
ax[1].set_ylabel("Percentage")


# In[68]:


mean_predictions = np.mean(prob_predictions, axis=0)
std_predictions = np.std(prob_predictions, axis=0)
mean_targets = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).mean()

labels = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).columns.values

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(x=labels, y=mean_predictions, ax=ax[0])
ax[0].set_xticklabels(labels=labels, rotation=90)
ax[0].set_ylabel("Mean predicted probability")
ax[0].set_title("Mean predicted probability per class over all samples")
sns.barplot(x=labels, y=std_predictions, ax=ax[1])
ax[1].set_xticklabels(labels=labels, rotation=90)
ax[1].set_ylabel("Standard deviation")
ax[1].set_title("Standard deviation per class over all samples");


# In[69]:


fig, ax = plt.subplots(1,1,figsize=(20,5))
sns.barplot(x=labels, y=mean_targets.values, ax=ax)
ax.set_xticklabels(labels=labels, rotation=90)
ax.set_ylabel("Percentage of hot (1)")
ax.set_title("Percentage of hot counts (ones) per target class")


# In[70]:


plt.figure(figsize=(20,5))
sns.distplot(CNN_prob_predictions["Cytosol"].values[0:-10], color="Blue")
plt.xlabel("Predicted probabilites of {}".format("Cytosol"))
plt.ylabel("Density")
plt.xlim([0,1])


# In[71]:


# add more evaluation metrics - F1 score

def base_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def f1_min(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.min(f1)

def f1_max(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.max(f1)

def f1_mean(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.mean(f1)

def f1_std(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.std(f1)


# In[72]:


# improve model suing a target wishlist

target_wishlist = ["Nucleoplasm", "Cytosol", "Plasma membrane"]


# In[73]:


# in contrast to the base data_generator we add a target wishlist to init
class data_generator2(data_generator):
    
    def __init__(self, list_IDs, labels, params, ipp, target_wishlist):
        super().__init__(list_IDs, labels, params, ipp)
        self.target_wishlist = target_wishlist
    
    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id==identifier][self.target_wishlist].values


# In[74]:


class track_history(Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


# In[22]:


class CNN2(CNN):
    
    def __init__(self, params, evaluation_metrics=[f1_mean, f1_std, f1_min, f1_max]):
        super().__init__(params)
        self.evaluation_metrics = evaluation_metrics
        
    def learn(self):
        self.history = track_history()
        return self.model.fit_generator(generator=self.train_gen, validation_data=self.test_gen, 
                                        epochs=self.params.n_epochs, use_multiprocessing=True, workers=8, 
                                        callbacks = [self.history])
    
    #add dropout
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                             kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))


# In[75]:


# increase number of epochs and decrease batch size
params = model_parameters(train_path, num_classes=len(target_wishlist), n_epochs=5, batch_size=64)
ipp = image_preprocessor(params)


# In[76]:


train_gen = data_generator2(partition['train'], train_labels, params, ipp, target_wishlist)
test_gen = data_generator2(partition['test'], train_labels, params, ipp, target_wishlist)
predict_gen = predict_generator(partition['test'], ipp, train_path)


# In[77]:


tipp = image_preprocessor(params)
submission_predict_gen = predict_generator(test_names, tipp, test_path)


# In[78]:


model = CNN2(params)
model.build_model()
model.compile_model()
model.set_generators(train_gen, test_gen)
epoch_history = model.learn()
prob_predictions = model.predict(predict_gen)

CNN2_prob_predictions = pd.DataFrame(prob_predictions, columns=target_wishlist)
# CNN2_prob_predictions.to_csv("CNN2_predictions.csv")
CNN2_losses = pd.DataFrame(epoch_history.history["loss"], columns=["train_loss"])
CNN2_losses["val_loss"] = epoch_history.history["val_loss"]
# CNN2_losses.to_csv("CNN2_losses.csv")
CNN2_batch_losses = pd.DataFrame(model.history.losses, columns=["batch_losses"])
# CNN2_batch_losses.to_csv("CNN2_batch_losses.csv")

CNN2_submission_prob_predictions = model.predict(submission_predict_gen)
CNN2_test_labels = test_labels.copy()
CNN2_test_labels.loc[:, target_wishlist] = CNN2_submission_prob_predictions
# CNN2_test_labels.to_csv("CNN2_submission_proba.csv")


# In[79]:


fig, ax = plt.subplots(2,1,figsize=(20,13))
ax[0].plot(np.arange(1,6), CNN2_losses["train_loss"].values, 'r--o', label="train_loss")
ax[0].plot(np.arange(1,6), CNN2_losses["val_loss"].values, 'g--o', label="validation_loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss evolution per epoch")
ax[0].legend()
ax[1].plot(CNN2_batch_losses.batch_losses.values, 'r-+', label="train_batch_losses")
ax[1].set_xlabel("Number of update steps in total")
ax[1].set_ylabel("Train loss")
ax[1].set_title("Train loss evolution per batch");


# In[80]:


fig, ax = plt.subplots(3,1,figsize=(25,15))
sns.distplot(CNN2_prob_predictions.values[:,0], color="Blue", ax=ax[0])
ax[0].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[0]))
ax[0].set_xlim([0,1])
sns.distplot(CNN2_prob_predictions.values[:,1], color="Red", ax=ax[1])
ax[1].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[1]))
ax[1].set_xlim([0,1])
sns.distplot(CNN2_prob_predictions.values[:,2], color="Purple", ax=ax[2])
ax[2].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[2]))
ax[2].set_xlim([0,1]);


# In[81]:


CNN2_test_labels.head(1)


# In[82]:


CNN2_prob_predictions.set_index(partition["test"], inplace=True)
CNN2_prob_predictions.head(5)


# In[83]:


validation_labels.set_index("Id", inplace=True)
validation_labels.head(5)


# In[84]:


fig, ax = plt.subplots(3,1,figsize=(20,24))
for n in range(len(target_wishlist)):
    sns.distplot(CNN2_prob_predictions.loc[validation_labels[target_wishlist[n]] == 1, target_wishlist[n]], 
                 color="Green", label="1-hot", ax=ax[n])
    sns.distplot(CNN2_prob_predictions.loc[validation_labels[target_wishlist[n]] == 0, target_wishlist[n]], 
                 color="Red", label="0-zero", ax=ax[n])
    ax[n].set_title(target_wishlist[n])
    ax[n].legend()


# In[23]:


th_cytosol = 0.3
th_plasma_membrane = 0.2
th_nucleoplasm = 0.4


# In[85]:


CNN2_submission = CNN2_test_labels.copy()
CNN2_submission.head(2)


# In[86]:


CNN2_submission["Nucleoplasm"] = np.where(CNN2_test_labels["Nucleoplasm"] >= th_nucleoplasm, 1, 0)
CNN2_submission["Cytosol"] = np.where(CNN2_test_labels["Cytosol"] >= th_cytosol, 1, 0)
CNN2_submission["Plasma membrane"] = np.where(CNN2_test_labels["Plasma membrane"] >= th_plasma_membrane, 1, 0)
CNN2_submission.head(5)


# In[24]:


def transform_to_target(row):
    target_list = []
    reverse_train_labels = dict((v,k) for k,v in label_names.items())
    for col in validation_labels.drop(["Target", "number_of_targets"], axis=1).columns:
        if row[col] == 1:
            target_list.append(str(reverse_train_labels[col]))
    if len(target_list) == 0:
        return str(0)
    return " ".join(target_list)


# In[87]:


CNN2_submission["Predicted"] = CNN2_submission.apply(lambda l: transform_to_target(l), axis=1)


# In[88]:


submission = CNN2_submission.loc[:, ["Id", "Predicted"]]
# submission.to_csv("CNN2_submission.csv", index=False)
submission.head()


# In[25]:


params = model_parameters(train_path, num_classes=len(target_wishlist), n_epochs=10, batch_size=128)
ipp = image_preprocessor(params)

train_gen = data_generator2(partition['train'], train_labels, params, ipp, target_wishlist)
test_gen = data_generator2(partition['test'], train_labels, params, ipp, target_wishlist)
predict_gen = predict_generator(partition['test'], ipp, train_path)


# In[26]:


model = CNN2(params)
model.build_model()
model.compile_model()
model.set_generators(train_gen, test_gen)
epoch_history = model.learn()
prob_predictions = model.predict(predict_gen)

CNN2_prob_predictions = pd.DataFrame(prob_predictions, columns=target_wishlist)
# CNN2_prob_predictions.to_csv("CNN2_prob_predictions.csv")
CNN2_losses = pd.DataFrame(epoch_history.history["loss"], columns=["train_loss"])
CNN2_losses["val_loss"] = epoch_history.history["val_loss"]
# CNN2_losses.to_csv("CNN2_losses.csv")
CNN2_batch_losses = pd.DataFrame(model.history.losses, columns=["batch_losses"])
# CNN2_batch_losses.to_csv("CNN2_batch_losses.csv")


# In[27]:


fig, ax = plt.subplots(2,1,figsize=(20,13))
ax[0].plot(np.arange(1,11), CNN2_losses["train_loss"].values, 'r--o', label="train_loss")
ax[0].plot(np.arange(1,11), CNN2_losses["val_loss"].values, 'g--o', label="validation_loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss evolution per epoch")
ax[0].legend()
ax[1].plot(CNN2_batch_losses.batch_losses.values, 'r-+', label="train_batch_losses")
ax[1].set_xlabel("Number of update steps in total")
ax[1].set_ylabel("Train loss")
ax[1].set_title("Train loss evolution per batch");


# In[28]:


fig, ax = plt.subplots(3,1,figsize=(25,15))
sns.distplot(CNN2_prob_predictions.values[:,0], color="Orange", ax=ax[0])
ax[0].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[0]))
ax[0].set_xlim([0,1])
sns.distplot(CNN2_prob_predictions.values[:,1], color="Purple", ax=ax[1])
ax[1].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[1]))
ax[1].set_xlim([0,1])
sns.distplot(CNN2_prob_predictions.values[:,2], color="Limegreen", ax=ax[2])
ax[2].set_xlabel("Predicted probabilites of {}".format(CNN2_prob_predictions.columns.values[2]))
ax[2].set_xlim([0,1]);

