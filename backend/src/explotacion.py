# -*- coding: utf-8 -*-
##------------------------------------------------------------------##
##------------- Carga de librerías y funciones----------------------##

##### Librerías

import pandas as pd
import numpy as np
import os
import time
import copy
import codecs
import re
import gc
from sklearn.feature_extraction.text import CountVectorizer
import gensim
import pickle
from scipy import spatial
from keras.models import load_model
import keras
from gensim.models import Word2Vec
import config

EMBEDDING_SIZE = 200


##### Funciones

##Distancias para reconocimiento de entidades
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in xrange(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in xrange(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition
    return d[lenstr1 - 1, lenstr2 - 1]


## Reconocimiento de entidad
def seleccion_entidad(entidad_list, ent_input):
    dist = []
    for ent in entidad_list:
        dist.append(damerau_levenshtein_distance(ent_input, ent))
    entidad = locate_min(dist, entidad_list)

    return entidad


## Aplicación de diccionarios:
def apply_dict(dat, diccionario):
    pattern = re.compile(r'\b(' + '|'.join(diccionario.keys()) + r')\b')
    tmp1 = dat
    dat = [pattern.sub(lambda x: diccionario[x.group()], item) for item in tmp1]
    return dat


## Lógica para el cálculo de distancias:
def evaluar_IR_w2v(sentence):
    ## RELEVANCIA TFIDF
    response = tfv.transform([sentence])

    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    terminos = []
    for col in response.nonzero()[1]:
        terminos.append([feature_names[col], response[0, col]])
        ##print feature_names[col], ' - ', response[0, col]

    terminos = pd.DataFrame(terminos)
    terminos.columns = ['TERMINO', 'VALOR']

    ## Fijamos el peso de las palabras no existentes en la tf-idf en el percentil 5

    terminos_tfidf = pd.concat([pd.DataFrame(feature_names), pd.DataFrame(tfv.idf_)], axis=1)
    terminos_tfidf.columns = ['TERMINO', 'VALOR']
    terminos_tfidf_ordenado = terminos_tfidf.sort_values(by=['VALOR'], ascending=False)

    peso_palabras_nuevas = np.percentile(terminos_tfidf['VALOR'], 0.01, axis=0)

    valor_0 = terminos['VALOR'][0]
    termino_0 = terminos_tfidf[terminos_tfidf['TERMINO'] == terminos['TERMINO'][0]].reset_index()['VALOR'][0]
    peso_nueva_normalizado = (peso_palabras_nuevas * valor_0) / termino_0

    words = sentence.split()

    for word in words:
        if word not in terminos['TERMINO'].tolist():
            terminos = terminos.append({'TERMINO': word, 'VALOR': peso_nueva_normalizado}, ignore_index=True)

    terminos = terminos.sort_values(by=['VALOR'], ascending=False)

    ## Fijamos número de términos relevantes
    numero_terminos = min(len(sentence.split(' ')), 5)  ## ¡¡¡¡Evaluar número óptimo de términos relevantes!!!!
    terminos_relavantes = terminos.head(numero_terminos)
    terminos_relavantes = terminos_relavantes.reset_index()

    ###### TOP3 DISTANCIA COSENO W2V

    ## Seleccionamos el top3 de distancias de coseno min de w2v:
    top3_dist_w2v = []
    for word in terminos_relavantes.TERMINO:
        palabras = []
        if word in w2v_model.wv.vocab:
            related = w2v_model.wv.most_similar_cosmul(positive=word)
            related = pd.DataFrame(related).head(3)  ## ¡¡¡¡Evaluar si coger top n o distancia menor que x!!!!
            for term in related[0]:
                palabras += [term]

        top3_dist_w2v = top3_dist_w2v + [palabras]
    p95 = 9.72548
    dist_w2v = pd.DataFrame(cod_list)
    for sinonimos in top3_dist_w2v:
        ## si el término existe en w2v asociamos el vecstor de mínimos
        if len(sinonimos) > 0:
            term = [sinonimo for sinonimo in sinonimos]
            palabras = [descripcion.split(" ") for descripcion in desc_list]
            dist_desc = [seleccion_entidad(palabra, termino) for palabra in palabras for termino in term]

            df_distancias = pd.DataFrame(np.asarray(pd.DataFrame(dist_desc)[0]).reshape((len(CIE_10), len(sinonimos))))
            minimo_fila = pd.DataFrame(df_distancias.min(axis=1))
            minimo_fila.columns = ['MIN']
            dist_w2v = pd.concat([dist_w2v, minimo_fila], axis=1)
        ## si el término no está en el w2v asociamos el p95 al término
        else:
            dist_p95 = pd.DataFrame(p95, index=range(len(CIE_10)), columns=range(1))
            dist_p95.columns = ['dist_p95']
            dist_w2v = pd.concat([dist_w2v, dist_p95], axis=1)

    del dist_w2v['CODIGO']
    dist_w2v.columns = list(range(len(top3_dist_w2v)))

    ## Ponderamos por la relevancia de los términos en la tf-idf
    epsilon = 0.001
    ##distancias = [distancia[0] for distancia in dist_desc]
    eps_dist = [(dist_w2v[indice] + epsilon) / terminos_relavantes['VALOR'][indice] for indice in
                range(0, len(terminos_relavantes))]
    df_distancias_eps = pd.DataFrame(np.asarray(pd.DataFrame(eps_dist)).T)

    ## Hallamos la distancia mínima como la suma de las columnas
    suma = pd.DataFrame(df_distancias_eps.sum(axis=1))
    suma.columns = ['SUMA']
    df_distancias_eps = pd.concat([df_distancias_eps, suma], axis=1)

    categoria = locate_min(suma['SUMA'], cod_list)[1][0]

    return categoria, suma


def evaluar_IR_dist(sentence):
    ## RELEVANCIA TFIDF
    response = tfv.transform([sentence])

    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

    terminos = []
    for col in response.nonzero()[1]:
        terminos.append([feature_names[col], response[0, col]])
        ##print feature_names[col], ' - ', response[0, col]

    terminos = pd.DataFrame(terminos)
    terminos.columns = ['TERMINO', 'VALOR']

    ## Fijamos el peso de las palabras no existentes en la tf-idf en el percentil 5

    terminos_tfidf = pd.concat([pd.DataFrame(feature_names), pd.DataFrame(tfv.idf_)], axis=1)
    terminos_tfidf.columns = ['TERMINO', 'VALOR']
    terminos_tfidf_ordenado = terminos_tfidf.sort_values(by=['VALOR'], ascending=False)

    peso_palabras_nuevas = np.percentile(terminos_tfidf['VALOR'], 0.01, axis=0)
    ##print(peso_palabras_nuevas)

    valor_0 = terminos['VALOR'][0]
    termino_0 = terminos_tfidf[terminos_tfidf['TERMINO'] == terminos['TERMINO'][0]].reset_index()['VALOR'][0]
    peso_nueva_normalizado = (peso_palabras_nuevas * valor_0) / termino_0

    words = sentence.split()

    for word in words:
        if word.decode('utf-8') not in terminos['TERMINO'].tolist():
            terminos = terminos.append({'TERMINO': word, 'VALOR': peso_nueva_normalizado}, ignore_index=True)

    terminos = terminos.sort_values(by=['VALOR'], ascending=False)

    ## Búsqueda:
    term = [termino for termino in terminos['TERMINO']]
    palabras = [descripcion.split(" ") for descripcion in desc_list]
    dist_desc = [seleccion_entidad(palabra, termino) for palabra in palabras for termino in term]

    df_distancias = pd.DataFrame(np.asarray(pd.DataFrame(dist_desc)[0]).reshape((len(CIE_10), len(terminos))))

    epsilon = 0.001
    distancias = [distancia[0] for distancia in dist_desc]
    eps_dist = [(df_distancias[indice] + epsilon) / terminos['VALOR'][indice] for indice in range(0, len(terminos))]

    df_distancias_eps = pd.DataFrame(np.asarray(pd.DataFrame(eps_dist)).T)

    ## Suma columnas
    suma = pd.DataFrame(df_distancias_eps.sum(axis=1))
    suma.columns = ['SUMA']
    df_distancias_eps = pd.concat([df_distancias_eps, suma], axis=1)

    categoria = locate_min(suma['SUMA'], cod_list)[1][0]

    return categoria, suma


## Correcciones
def correcciones(desc_list):
    df_desc_list = pd.DataFrame(desc_list)

    df_desc_list = df_desc_list.dropna()

    ## Aplicamos correcciones básico


    df_desc_list["DOC_CONTENT2"] = df_desc_list["DESCRIPCION"].str.lower()
    df_desc_list["DOC_CONTENT3"] = df_desc_list["DOC_CONTENT2"].apply(
        lambda x: re.sub(r'[\'\"]', r'', x.encode('UTF-8')))
    df_desc_list["DOC_CONTENT4"] = df_desc_list["DOC_CONTENT3"].str.replace(r"\b(\\xa0)\b", r"")
    df_desc_list["DOC_CONTENT5"] = df_desc_list["DOC_CONTENT4"].str.replace(r"\b(\\n)\b", r"")
    df_desc_list["DOC_CONTENT6"] = df_desc_list["DOC_CONTENT5"].apply(
        lambda x: re.sub(r'[^a-zA-ZÁáÉéÍíÓóÚúÜüÑñ\s]', r' ', x))
    df_desc_list["DOC_CONTENT7"] = df_desc_list["DOC_CONTENT6"].apply(lambda x: re.sub(r'\s+', r' ', x)).str.strip()

    ##Quitamos los vacíos de nuevo
    df_desc_list = df_desc_list.loc[df_desc_list["DOC_CONTENT7"] != ""]

    ## Copus final
    df_desc_list = df_desc_list[["DOC_CONTENT7"]]
    df_desc_list.columns = ["DESCRIPCION"]

    dic_lex = \
    pd.read_csv(config.DIC_LEX_IR, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
        orient='dict')['CORRECCION']
    df_desc_list["DESCRIPCION"] = apply_dict(df_desc_list["DESCRIPCION"], dic_lex)

    dic_corr = \
    pd.read_csv(config.DIC_CORR_2, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
        orient='dict')['CORRECCION']
    df_desc_list["DESCRIPCION"] = apply_dict(df_desc_list["DESCRIPCION"], dic_corr)

    dic_ngramas = \
    pd.read_csv(config.DICT_NGRAMMA, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
        orient='dict')['CORRECCION']
    df_desc_list["DESCRIPCION"] = apply_dict(df_desc_list["DESCRIPCION"], dic_ngramas)

    desc_list = desc_list = df_desc_list['DESCRIPCION']

    return desc_list


def locate_min3(df, entidad_list):
    df.columns = ['COL1']

    a = df['COL1'].tolist()
    a1 = np.array(a)
    indices = a1.argsort()
    indices = indices[a1[indices] != -5]

    min3 = pd.DataFrame(columns=['VALUE', 'CODE'])
    for i in indices[:3]:
        min3 = min3.append({'VALUE': df['COL1'][i], 'CODE': entidad_list['CODIGO'][i]}, ignore_index=True)

    min3 = min3.sort_values(by=['VALUE'], ascending=True)

    return min3


def tratamiento_frase(sentence):
    sentence = pd.DataFrame(columns=["DOC_ID", "DOC_CONTENT"])
    sentence = sentence.append({'DOC_ID': 'XXXX', 'DOC_CONTENT': original_sentence}, ignore_index=True)

    corpus_sentence = copy.deepcopy(sentence)

    ##Quitamos los vacíos
    corpus_sentence = corpus_sentence.dropna()

    ##Aplicamos tratamiento básico
    corpus_sentence["DOC_CONTENT2"] = corpus_sentence["DOC_CONTENT"].str.lower()
    corpus_sentence["DOC_CONTENT3"] = corpus_sentence["DOC_CONTENT2"].apply(lambda x: re.sub(r'[\'\"]', r'', x))
    corpus_sentence["DOC_CONTENT4"] = corpus_sentence["DOC_CONTENT3"].str.replace(r"\b(\\xa0)\b", r"")
    corpus_sentence["DOC_CONTENT5"] = corpus_sentence["DOC_CONTENT4"].str.replace(r"\b(\\n)\b", r"")
    corpus_sentence["DOC_CONTENT6"] = corpus_sentence["DOC_CONTENT5"].apply(
        lambda x: re.sub(r'[^a-zA-ZÁáÉéÍíÓóÚúÜüÑñ\s]', r' ', x))
    corpus_sentence["DOC_CONTENT8"] = corpus_sentence["DOC_CONTENT6"].apply(
        lambda x: re.sub(r'\s+', r' ', x)).str.strip()

    ##Quitamos los vacíos de nuevo
    corpus_sentence = corpus_sentence.loc[corpus_sentence["DOC_CONTENT8"] != ""]

    corpus_sentence["DOC_CONTENT10"] = corpus_sentence["DOC_CONTENT8"].apply(
        lambda x: re.sub(r'\s+', r' ', x)).str.strip()

    ##Quitamos los vacíos de nuevo
    corpus_sentence = corpus_sentence.loc[corpus_sentence["DOC_CONTENT10"] != ""]

    ##Guardamos el corpus_sentence limpio
    clean_corpus_sentence = corpus_sentence[["DOC_CONTENT10"]]
    clean_corpus_sentence.columns = ["DOC_CONTENT"]

    return clean_corpus_sentence['DOC_CONTENT'][0]


def locate_min(a, entidad_list):
    smallest = min(np.float64(a))
    indices = [entidad_list[index] for index, element in enumerate(np.float64(a)) if
               np.float64(smallest) == np.float64(element)]

    return smallest, indices


def locate_max_float(a, entidad_list):
    smallest = max(np.float64(a))
    indices = [entidad_list[index] for index, element in enumerate(np.float64(a)) if
               np.float64(smallest) == np.float64(element)]

    return smallest, indices


def evaluar_IR_LDA(sentence):
    sentence = sentence.lower()
    sentence = re.sub('á', 'a', re.sub('é', 'e', re.sub('í', 'i', re.sub('ó', 'o', re.sub('ú', 'u', sentence)))))
    test = [sentence]
    test_texts = [[token for token in text.split(' ')] for text in test]
    test_corpus = [dictionary.doc2bow(text) for text in test_texts]

    vec_lda = lda[test_corpus]
    sims = index[vec_lda]
    df_sims = pd.DataFrame(sims[0])

    categoria = locate_max_float(df_sims, cod_list)[1][0]
    return categoria, df_sims


def explotacion_IR(original_sentence):
    original_sentence = original_sentence.lower()
    original_sentence = re.sub('à', 'a', re.sub('è', 'e', re.sub('ì', 'i', re.sub('ò', 'o', re.sub('ù', 'u',
                                                                                                   original_sentence)))))

    ## Explotamos modelos
    sentence = original_sentence.decode('utf-8')
    sentence = pd.DataFrame([sentence], columns=['DESCRIPCION'])
    sentence = correcciones(sentence)[0]

    ## Llamada IR distancias
    resultado_dist, vect_distancias = evaluar_IR_dist(sentence)

    ## Llamada IR w2v
    resultado_w2v, vect_w2v = evaluar_IR_w2v(sentence)

    ## Llamada IR LSI
    resultado_LDA, vect_LDA = evaluar_IR_LDA(original_sentence)

    vectores = [vect_distancias, vect_w2v, vect_LDA]

    ##print("-----------------------------------------------------------")
    ##print("Resultado IR por distancias:   ")
    ##print("Frase original:     ", original_sentence)
    ##print("Categoría asignada: ", resultado_dist)
    ##print("-----------------------------------------------------------")
    ##print("Resultado IR por word2vect:   ")
    ##print("Frase original:     ", original_sentence)
    ##print("Categoría asignada: ", resultado_w2v)
    ##print("-----------------------------------------------------------")
    ##print("Resultado IR por LDA:   ")
    ##print("Frase original:     ", original_sentence)
    ##print("Categoría asignada: ", resultado_LDA)
    ##print("-----------------------------------------------------------")


    # Cálculo score
    score = model_IR.predict(
        [np.asarray(vect_distancias.transpose()), np.asarray(vect_w2v.transpose()), np.asarray(vect_LDA.transpose())])
    ##print(score)

    categorias = [resultado_dist, resultado_w2v, resultado_LDA]
    resultado = categorias[np.argmax(score[0])]
    ##print(np.argmax(score[0]))
    ##print(resultado)

    df = pd.concat([cod_agrup_list, desc_agrup_list, vectores[np.argmax(score[0])]], axis=1)
    df.columns = ['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'DIST']

    if np.argmax(score[0]) == 2:
        result = df.groupby('CODIGO_AGRUP', group_keys=False).apply(max)
        result = result.sort_values(by=['DIST'], ascending=False)
        result = pd.DataFrame(np.array(result[['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'DIST']]))
        result.columns = ['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'DIST']
    else:
        result = df.groupby('CODIGO_AGRUP', group_keys=True).apply(min)
        result = result.sort_values(by=['DIST'], ascending=True)
        result = pd.DataFrame(np.array(result[['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'DIST']]))
        result.columns = ['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'DIST']

    ##result= result[['CODIGO_AGRUP','DESCRIPCION_AGRUP']]

    return result


# Carga de funciones
# Tratamiento de texto:
def tratamiento(sentence):
    # Quitar acentos:
    sentence = re.sub('Á', 'A', re.sub('É', 'E', re.sub('Í', 'I', re.sub('Ó', 'O', re.sub('Ú', 'U', re.sub('á', 'a',
                                                                                                           re.sub('é',
                                                                                                                  'e',
                                                                                                                  re.sub(
                                                                                                                      'í',
                                                                                                                      'i',
                                                                                                                      re.sub(
                                                                                                                          'ó',
                                                                                                                          'o',
                                                                                                                          re.sub(
                                                                                                                              'ú',
                                                                                                                              'u',
                                                                                                                              sentence))))))))))
    # Quitar caracteres extraños
    sentence = re.sub('[^a-zA-Zñ<> ]', '', sentence)
    # Dejar sólo un espacio entre palabras
    sentence = " ".join(sentence.split())
    # Poner todo en minuscula y sin espacios al principio y al final
    return sentence.strip().lower()


# Construcción del vocabulario
def get_vocab(sents, maxvocab=25000, stoplist=[], verbose=False):
    # get vocab list
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word)

    counts = Counter(vocab)  # get counts of each word
    vocab_set = list(set(vocab))  # get unique vocab list
    sorted_vocab = sorted(vocab_set, key=lambda x: counts[x], reverse=True)  # sort by counts
    sorted_vocab = [i for i in sorted_vocab if i not in stoplist]
    if verbose:
        print("\ntotal vocab size:", len(sorted_vocab), '\n')
    sorted_vocab = sorted_vocab[:maxvocab - 2]
    if verbose:
        print("\ntrunc vocab size:", len(sorted_vocab), '\n')
    vocab_dict = {k: v + 1 for v, k in enumerate(sorted_vocab)}
    vocab_dict['UNK'] = len(sorted_vocab) + 1
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
def explotacion_LSTM(sentence):
    texto_tratado = tratamiento(sentence)
    ## Aplicación de los diccionarios al contenido (correccion + lexematizacion)
    df_texto_tratado = pd.DataFrame([texto_tratado])
    df_texto_tratado.columns = ['DOC_CONTENT']
    df_texto_tratado["DOC_CONTENT"] = apply_dict(df_texto_tratado["DOC_CONTENT"], dic_corr)
    df_texto_tratado["DOC_CONTENT"] = apply_dict(df_texto_tratado["DOC_CONTENT"], dic_lex)

    ## Construcción diccionario términos corregidos y lexematizados:
    texto_corregido = np.array(df_texto_tratado['DOC_CONTENT'].tolist())
    # allwords = [s.split(' ') for s in texto_corregido]
    # df_texto_corregido=pd.DataFrame(texto_corregido)

    ## Split por palabra
    array_textos = [texto_corregido[0].split(" ")]

    ## Word2idx
    array_textos_new = []

    a = [0] * EMBEDDING_SIZE
    for j in range(0, len(array_textos[0])):
        if array_textos[0][j] in word2idx.keys():
            a[j] = word2idx[array_textos[0][j]]
    X_tokens = np.array([a])

    # Evaluación frase
    score = model.predict(X_tokens)
    dic_cat = pd.read_csv(config.DIC_CAT, sep=";", encoding='utf-8')
    dic_cat['SCORE'] = score[0]

    dic_cat = dic_cat[['CODIGO_CIE', 'NOMBRE', 'SCORE']]
    dic_cat.columns = ['CODIGO_AGRUP', 'DESCRIPCION_AGRUP', 'SCORE']
    dic_cat = dic_cat.sort_values(by=['SCORE'], ascending=False)

    ##dic_cat= dic_cat[['CODIGO_AGRUP','DESCRIPCION_AGRUP']]


    return dic_cat


##------------------------------------------------------------------##

##------------------------------------------------------------------##
##---------------------- Carga de datos ----------------------------##


#### LSTM

## Carga de modelos LSTM
model = load_model(config.LSTM)

## Carga de diccionarios
# Token
pkl_file = open(config.WORD2IDX, 'rb')
word2idx = pickle.load(pkl_file)
# Diccionario correcciones
dic_corr = pd.read_csv(config.DIC_CORR_1, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
    orient='dict')['CORRECCION']
# Diccionario lexematización
dic_lex = \
pd.read_csv(config.DIC_LEX, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(orient='dict')[
    'CORRECCION']
# Diccionario categorías
dic_cat = pd.read_csv(config.DIC_CAT, sep=";", encoding='utf-8')

#### IR:

#### Carga tfidf
pkl_file = open(config.FEATURE_NAMES, 'rb')
feature_names = pickle.load(pkl_file)

pkl_file = open(config.TFV, 'rb')
tfv = pickle.load(pkl_file)

## Carga descripciones CIE_10
CIE_10 = pd.read_csv(config.MAESTRA_CIE10, sep=';', encoding='latin1')
desc_list = CIE_10['DESCRIPCION']
cod_list = CIE_10['CODIGO']
cod_agrup_list = CIE_10['CODIGO_AGRUP']
desc_agrup_list = CIE_10['DESCRIPCION_AGRUP']

## Cargamos modelo w2v
w2v_model = gensim.models.Word2Vec.load(config.W2V_IR)

## Cargamos modelo LDA
dictionary = gensim.corpora.Dictionary.load(config.DICTIONARY)
lda = gensim.models.ldamodel.LdaModel.load(config.LDA)
index = gensim.similarities.MatrixSimilarity.load(config.INDEX)

## Cargamos RED para rank
model_IR = load_model(config.MODELO_RANK)

##------------------------------------------------------------------##


##------------------------------------------------------------------##
##---------------------- Explotación -------------------------------##

##original_sentence = 'sndrome coronrio agdo con eleación del st' ## ejemplo dist
##original_sentence = 'varices en pierna derecha con ulcera de localizacion no especificada' ## ejemplo w2v
##original_sentence = 'Insuficiencia venosa (cronica) (periferica)' ## ejemplo LDA
original_sentence = 'síndrome coronària aguda amb elevació del st'  ## ejemplo catalán

## Resultado clasificador
result_LSTM = explotacion_LSTM(original_sentence)
print result_LSTM
## Resultado IR:
result_IR = explotacion_IR(original_sentence)
print result_IR

##------------------------------------------------------------------##
