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

def predict_path(image, model):
    aux = image[50:220,:,:]
    dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
    aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((1,dim[1],dim[0],1))
    pred = model.predict(aux)
    print(pred)
    return(np.argmax(pred))

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

@st.cache(suppress_st_warning=True, allow_output_mutation=True, ttl = 3600)
def read_dataset():

    datas = []
    label = []
    ori = []
    
    model = models.model_from_json(open(wd+"model_bombacav.json","r").read())
    model.load_weights("weight_bombacav.h5")
    
    for direc in folders:
        name_dir = wd+direc
        images = os.listdir(name_dir)
        for i in images:
            ori.append(np.array(mpimg.imread(name_dir +'/' + i, )[:,:,:])) 
            aux = np.array(mpimg.imread(name_dir +'/' + i, )[50:220,:,:])
            dim = int(aux.shape[1]/dim_division), int(aux.shape[0]/dim_division)
            aux = ((np.array(Image.fromarray(aux).convert('L').resize(dim)) - np.min(aux))/(np.max(aux) - np.min(aux))).reshape((dim[1],dim[0],1))
            label.append(np.where(np.array(folders) == direc)[0][0])
            datas.append(aux)
    #        print(dim)
        print (direc)
    
    datas = np.asarray(datas)
    return datas, ori, model


    
#%%
aux = np.array(mpimg.imread(wd + 'marca-topo.jpg'))
#aux.resize((200,75))
st.image(aux, width = 100)

st.title("""Laborat??rio de An??lise de Vibra????es - Senai Cimatec""")

st.header("""Diagn??stico de cavita????o em bombas""")

st.subheader("""Modelo de Intelig??ncia Artificial""")

st.write("""O modelo foi construido baseado nos dados de uma bancada de teste de cavita????o de bomba, os dados foram divididos em 3 n??veis de cavita????o distintos simulados a partir das v??lvulas destacadas, dados de vibra????o foram coletados utilizando o equipamento e software da TEKNIKAO (https://www.teknikao.com.br/) o qual emite gr??ficos resultante de uma transformada r??pida de fourier""")
st.write("""Confira abaixo como o n??vel de cavita????o altera o gr??fico para os dados registrados: """)

st.image(np.array(mpimg.imread(wd + 'Bancada.jpeg')))


fm = ["Sem cavita????o", "Pouca cavita????o", "Muita cavita????o"]

st.write("Veja voc?? mesmo como a cavita????o altera o comportamento do gr??fico...")
aux = st.radio('Modo de falha:', fm)

dataset, orig, model_l = read_dataset()


ind = np.where([m==aux for m in fm])[0][0]
fig = plot_spectra(ind)

    
st.write('A ideia ?? que o modelo de aprendizagem de m??quina, a partir de um conjunto de imagens de treinamento, possa observar essas altera????es no comportamento do gr??fico . O modelo ?? uma rede neural convolucional 2d de processamento de imagem, cujo treinamento foi feito utilizando ' + str(int(dataset.shape[0]*0.99)) + ' imagens e o teste foi executado utilizando ' + str(int(dataset.shape[0]*0.33)) +' imagens')

st.write('Ap??s o treinamento, ?? necess??rio avaliar os resultados para um conjunto de dados separado dos que a IA usou para aprender, Os resultados das previs??es no conjunto de teste s??o apresentados na matriz de confus??o. Note que nossa rede erra apenas 1 dos dados de testes :)')

aux = np.array(mpimg.imread(wd + 'Conf_matrix.jpg'))

st.image(aux)


st.subheader("Voc?? pode tamb??m acrescentar novas imagens e o modelo ir?? realizar novas previs??es")
st.write("Envie uma nova imagem para diagn??stico")
img_file_buffer = st.file_uploader("Envie")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    image = np.array(image)
    st.image(image)

    aux = fm[predict_path(image, model_l)]
    st.subheader("O modelo previu o seguinte resultado para a imagem: ")
    if aux == "Sem cavita????o":
        st.image(np.array(mpimg.imread(wd+'icons/Bom_cav.png')))
    if aux == "Pouca cavita????o":
        st.image(np.array(mpimg.imread(wd+'icons/Pouco_cav.png')))
    if aux == "Muita cavita????o":
        st.image(np.array(mpimg.imread(wd+'icons/critico.png')))

st.subheader("Podemos te enviar uma imagem de exemplo para testar a previs??o do modelo")
#agree = st.checkbox("Manda")

if st.button("Envia"):
    exemplo = random.choice(orig)
    st.image(exemplo)

#%%
#%% 
#!streamlit run "app.py"
