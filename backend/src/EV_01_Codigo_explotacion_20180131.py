# -*- coding: utf-8 -*-

##-----------------------------------------------------------------------------------##
##----------------------------- Requerimientos --------------------------------------##

##pip install keras==2.0.6

##-----------------------------------------------------------------------------------##



##-----------------------------------------------------------------------------------##
##-------------------------- Funciones y librerías ----------------------------------##

# Carga de librerias 
import numpy as np
import keras
import pandas as pd
import re
import os
import gensim
import pickle 
from keras.models import load_model
from gensim.models import Word2Vec
import pdb

EMBEDDING_SIZE = 200


# Carga de funciones 
# Tratamiento de texto:
def  tratamiento(sentence):
    #Quitar acentos:
    sentence = re.sub('Á', 'A',re.sub('É', 'E',re.sub('Í', 'I',re.sub('Ó', 'O',re.sub('Ú', 'U', re.sub('á', 'a',re.sub('é', 'e',re.sub('í', 'i',re.sub('ó', 'o',re.sub('ú', 'u', sentence))))))))))
    #Quitar caracteres extraños
    sentence = re.sub('[^a-zA-Zñ<> ]', '', sentence) 
    #Dejar sólo un espacio entre palabras
    sentence=" ".join( sentence.split() ) 
    #Poner todo en minuscula y sin espacios al principio y al final  
    return sentence.strip().lower()

# Construcción del vocabulario
def get_vocab(sents, maxvocab=25000, stoplist=[], verbose=False):

    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab) # get counts of each word
    vocab_set = list(set(vocab)) # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: counts[x], reverse=True) # sort by counts
    sorted_vocab = [i for i in sorted_vocab if i not in stoplist]
    if verbose:
        print("\ntotal vocab size:", len(sorted_vocab), '\n')
    sorted_vocab = sorted_vocab[:maxvocab-2]
    if verbose:
        print("\ntrunc vocab size:", len(sorted_vocab), '\n')
    vocab_dict = {k: v+1 for v, k in enumerate(sorted_vocab)}
    vocab_dict['UNK'] = len(sorted_vocab)+1
    vocab_dict['PAD'] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict

# Aplicación de diccionarios al contenido
def apply_dict(dat, diccionario):
    pattern = re.compile(r'\b(' + '|'.join(diccionario.keys()) + r')\b')
    tmp1 = dat
    dat = [pattern.sub(lambda x: diccionario[x.group()], item) for item in tmp1]
    return dat


## Tratamiento completo
def explotacion(sentence):    
    
    texto_tratado =tratamiento(sentence)
    ## Aplicación de los diccionarios al contenido (correccion + lexematizacion)
    df_texto_tratado = pd.DataFrame([texto_tratado])
    df_texto_tratado.columns = ['DOC_CONTENT']
    df_texto_tratado["DOC_CONTENT"] = apply_dict(df_texto_tratado["DOC_CONTENT"], dic_corr)
    print(df_texto_tratado["DOC_CONTENT"])
    df_texto_tratado["DOC_CONTENT"] = apply_dict(df_texto_tratado["DOC_CONTENT"], dic_lex)
    print (df_texto_tratado["DOC_CONTENT"])
    ## Construcción diccionario términos corregidos y lexematizados:
    texto_corregido = np.array(df_texto_tratado['DOC_CONTENT'].tolist())
    #allwords = [s.split(' ') for s in texto_corregido]
    #df_texto_corregido=pd.DataFrame(texto_corregido)

    ## Split por palabra
    array_textos =[texto_corregido[0].split(" ")]

    ## Word2idx
    array_textos_new = []

    a = [0] * EMBEDDING_SIZE
    for j in range (0, len(array_textos[0])):
        if  array_textos[0][j] in word2idx.keys():
            a[j] = word2idx[array_textos[0][j]]    
    X_tokens = np.array([a])

    # Evaluación frase
    score = model.predict(X_tokens)
    dic_cat['SCORE']=score[0]
    
    
    return dic_cat

##-----------------------------------------------------------------------------------##



##-----------------------------------------------------------------------------------##
##---------------------------- Carga de ficheros ------------------------------------##

## Carga de modelos:
model = load_model(os.path.join(config.MODEL, 'BI_LSTM_50_lex_corr.h5'))

## Carga de diccionarios
# Token
pkl_file = open(os.path.join(config.MODEL, 'word2idx.pkl'), 'rb')
word2idx = pickle.load(pkl_file) 
# Diccionario correcciones
dic_corr = pd.read_csv(os.path.join(config.DICTS, 'PD_01_Diccionario_correc.csv'), sep = ";", index_col = 0, converters = {'TEXTO':str, 'CORRECCION':str}).to_dict(orient='dict')['CORRECCION']
# Diccionario lexematización
dic_lex = pd.read_csv(os.path.join(config.DICTS, 'PD_02_Diccionario_lex.csv'), sep = ";", index_col = 0, converters = {'TEXTO':str, 'CORRECCION':str}).to_dict(orient='dict')['CORRECCION']
# Diccionario categorías
dic_cat = pd.read_csv(os.path.join(config.DICTS, 'CD_01_Diccionario_categorias.csv'),sep = ";", encoding = 'utf-8')

##-----------------------------------------------------------------------------------##



##-----------------------------------------------------------------------------------##
##---------------------------- Explotación modelo -----------------------------------##

sentence = 'Enfermedad'
salida_modelo = explotacion(sentence)

print(salida_modelo)

##-----------------------------------------------------------------------------------##
