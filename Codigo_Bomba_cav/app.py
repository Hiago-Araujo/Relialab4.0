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
from keras import models

#%%
wd = "/app/relialab4.0/Codigo_Bomba_cav/"


#%%



def plot_spectra(ind):
    name_dir = wd+folders[ind]
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
def button_manda():
    return st.button("Manda")

@st.cache(suppress_st_warning=True)
def read_dataset():

    dataset = []
    label = []
    orig = []
    
    for direc in folders:
        name_dir = wd+direc
        images = os.listdir(name_dir)
        for i in images:
            orig.append(np.array(mpimg.imread(name_dir +'/' + i, )[:,:,:])) 
            aux = np.array(mpimg.imread(name_dir +'/' + i, )[50:220,:,:])
            dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
            aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((dim[1],dim[0],1))
            label.append(np.where(np.array(folders) == direc)[0][0])
            dataset.append(aux)
    #        print(dim)
        print (direc)
    
    dataset = np.asarray(dataset)
    return dataset, orig


    
#%%
st.title("""Lab Preditiva 4.0""")

st.header("""Diagnóstico de bomba de cavitação""")

st.subheader("""Modelo de diagnóstico de bomba de cavitação""")

st.write("""O modelo foi construido baseado nos dados de uma bancada de teste de cavitação de bomba, os dados foram divididos em 3 níveis de cavitação distintos influenciado por válvulas, dados de vibração foram coletados utilizando o equipamento e software da TEKNIKAO o qual emite gráficos de espectro de frequência apresentado abaixo""")

st.write("""\n\n\ninserir foto da bancada\n\n\n""")

fm = ["Sem cavitação", "Pouca cavitação", "Muita cavitação"]

aux = st.radio('Modo de falha:', fm)

dataset, orig = read_dataset()


ind = np.where([m==aux for m in fm])[0][0]
fig = plot_spectra(ind)

    
st.write('O modelo foi treinado utilizando ' + str(int(dataset.shape[0]*0.8)) + ' imagens de treino e o teste foi executado utilizando' + str(int(dataset.shape[0]*0.2)) +' imagens de teste')

st.write('Os resultados das previsões no conjunto de teste são: ')

aux = np.array(mpimg.imread(wd + 'Conf_matrix.jpg'))

st.image(aux)


st.write("Envie uma nova imagem para diagnóstico")
img_file_buffer = st.file_uploader("Envie")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = np.array(image)

    model_l = models.model_from_json(open(wd+"model_bombacav.json","r").read())
    model_l.load_weights("weight_bombacav.h5")

    def predict_path(image, model):
        aux = image[50:220,:,:]
        st.image(aux)
        dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
        aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((1,dim[1],dim[0],1))
        pred = model.predict(aux)
        print(pred)
        return(np.argmax(pred))

    aux = fm[predict_path(image, model_l)]
    st.subheader("O modelo previu o seguinte resultado para a imagem: " + str(aux))

    st.subheader("Nova Classificação: " + str(aux))

    if aux == "Sem cavitação":
        st.image(np.array(mpimg.imread(wd+'icons/Bom_cav.png')))
    if aux == "Pouca cavitação":
        st.image(np.array(mpimg.imread(wd+'icons/Pouco_cav.png')))
    if aux == "Muita cavitação":
        st.image(np.array(mpimg.imread(wd+'icons/critico.png')))
    
st.subheader("Imagem de exemplo, baixe e envie para testar a previsão do modelo")
agree = st.checkbox("Manda")

if agree:
    exemplo = random.choice(orig)
    st.image(exemplo)
    st.text("Qual o seu palpite?")
    selec = st.selectbox("?", [''] + fm)
    
    if not selec == '':
        st.text("a previsão correta é...")

        aux = fm[predict_path(image, model_l)]

        if aux == "Sem cavitação":
            st.image(np.array(mpimg.imread(wd+'icons/Bom_cav.png')))
        if aux == "Pouca cavitação":
            st.image(np.array(mpimg.imread(wd+'icons/Pouco_cav.png')))
        if aux == "Muita cavitação":
            st.image(np.array(mpimg.imread(wd+'icons/critico.png')))


#%%
#%% 
#!streamlit run "app.py"
