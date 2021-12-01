# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:44:34 2021

@author: Hiago Araujo
"""
"""# Load Data"""
#Lendo as bibliotecas utilizadas:
#Bibliotecas de leituras, processamento e apresentação de dados
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg

#Biblioteca de redes neurais (keras)
import keras
from keras import models
from keras import layers
from keras import backend as K
from keras.models import Model
from keras import utils as np_utils
#Importa as bibliotecas utilizadas
import sklearn as skl
import scipy.misc as misc

import tensorflow as tf
import imageio
import sklearn.model_selection as model_selection
from sklearn.preprocessing import LabelEncoder

from PIL import Image

#%% Leitura de dados

wd = "C:/Users/Hiago Araujo/OneDrive/Desktop/Relialab/Codigo_Bomba_cav/"

#wd = "/gdrive/MyDrive/lab/Relialab/Codigo_Bomba_cav/"

folders = ["health_spectra", "fm1_spectra", "fm2_spectra"]

dataset = []
label = []
dim_division = 2

for direc in folders:
    name_dir = wd + direc
    images = os.listdir(name_dir)
    for i in images:
        aux = np.array(mpimg.imread(name_dir +'/' + i, )[50:220,:,:])
        dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
        aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((dim[1],dim[0],1))
        label.append(np.where(np.array(folders) == direc)[0][0])
        dataset.append(aux)
#        print(dim)
    print (direc)

dataset = np.asarray(dataset)

dataset.shape

#%%
dataset = np.asarray(dataset)
#%%
#%%
plt.imshow(dataset[0,:,:,0],cmap = 'gray')
dim
dataset[0].shape

#%%
xTrain, xTest, yTrain, yTest = model_selection.train_test_split (dataset, label, test_size=0.33, random_state=4)
yTrain_categorical = keras.utils.np_utils.to_categorical(yTrain)
yTest_categorical = keras.utils.np_utils.to_categorical(yTest)

xTrain = np.asarray(xTrain)

xTrain.shape

#%%
#Rede neural de classificação

model = models.Sequential()
model.add(layers.Conv2D(filters = 3, kernel_size = (4,4), activation = "relu", input_shape = dataset[0].shape))
model.add(layers.Conv2D(filters=3, kernel_size = (6,6), activation="relu"))
model.add(layers.Conv2D(filters=3, kernel_size = (10,10), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(256,activation="relu"))   
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.7))
model.add(layers.Dense(16,activation="relu"))

model.add(layers.Dense(3,activation="softmax"))


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

#%%
model.fit(xTrain, yTrain_categorical, batch_size=20, epochs=40)
#%%

c_matrix = np.array(skl.metrics.confusion_matrix(yTest, np.argmax(model.predict(xTest), axis = 1)))

df_cm = pd.DataFrame(c_matrix, index = [['Bom', 'Cavitado', 'Crítico']],
                  columns = [i for i in ['Bom', 'Cavitado', 'Crítico']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap="RdYlGn")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(wd + "Conf_matrix.jpg")

plt.show()

#%%
file = model.to_json()

with open(wd+"model_bombacav.json", "w") as json_file:
    json_file.write(file)