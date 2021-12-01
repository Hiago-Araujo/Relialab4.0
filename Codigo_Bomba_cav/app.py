# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:57:17 2021

@author: Hiago Araujo
"""

import streamlit as st
from io import StringIO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random

#%%
wd = "Codigo_Bomba_cav/"
os.chdir(wd)


#%%



def plot_spectra(ind):
    name_dir = folders[ind]
    images = os.listdir(name_dir)
    
    fig=plt.figure(figsize = (10,5))
    
    for j in range(9):
        i = random.choice(images)
        aux = np.array(mpimg.imread(name_dir +'/' + i, )[50:220,:,:])
        dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
        aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((dim[1],dim[0],1))
        fig.add_subplot(3,3,j+1)
        plt.imshow(aux, cmap = 'gray')
    
    st.pyplot(fig)
    return(fig)


folders = ["health_spectra", "fm1_spectra", "fm2_spectra"]

dim_division = 2
    
@st.cache(suppress_st_warning=True)
def read_dataset():

    dataset = []
    label = []
    
    for direc in folders:
        name_dir = direc
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
    st.write(dataset.shape)
    return dataset


    
#%%
st.title("""Lab Preditiva 4.0""")
st.write(str(os.getcwd()))
st.header("""Diagnóstico de bomba de cavitação""")

st.subheader("""Modelo de diagnóstico de bomba de cavitação""")

st.write("""O modelo foi construido baseado nos dados de uma bancada de teste de cavitação de bomba, os dados foram divididos em 3 níveis de cavitação distintos influenciado por válvulas, dados de vibração foram coletados utilizando o equipamento e software da TEKNIKAO o qual emite gráficos de espectro de frequência apresentado abaixo""")

st.write("""\n\n\ninserir foto da bancada\n\n\n""")

fm = ["Sem cavitação", "Pouca cavitação", "Muita cavitação"]

aux = st.radio('Modo de falha:', fm)

dataset = read_dataset()


ind = np.where([m==aux for m in fm])[0][0]
fig = plot_spectra(ind)

    
st.write('O modelo foi treinado utilizando ' + str(int(dataset.shape[0]*0.8)) + ' imagens de treino e o teste foi executado utilizando' + str(int(dataset.shape[0]*0.2)) +' imagens de teste')

st.write('Os resultados das previsões no conjunto de teste são: ')

aux = np.array(mpimg.imread('Conf_matrix.jpg'))

st.image(aux)

#%%
#%% 
#!streamlit run "app.py"
