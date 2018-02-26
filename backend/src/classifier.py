# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import keras
import os
import gensim
import re
import pickle
import unicodedata
import config
from keras.models import load_model
from gensim.models import Word2Vec
import tensorflow as tf
import pdb
from datetime import datetime
import time
import logging
import json
import math


class Classifier(object):
    def __init__(self):
        self.embedding_size = 200

        self.model_LSTM = load_model(config.LSTM)
        self.graph = tf.get_default_graph()

        self.pkl_file_1 = open(config.FEATURE_NAMES, 'rb')
        self.feature_names = pickle.load(self.pkl_file_1)

        self.pkl_file_2 = open(config.TFV, 'rb')
        self.tfv = pickle.load(self.pkl_file_2)

        self.CIE_10 = pd.read_csv(config.MAESTRA_CIE10, sep=';', encoding='latin1')
        self.desc_list = self.CIE_10['DESCRIPCION']
        self.cod_list = self.CIE_10['CODIGO']
        self.cod_agrup_list = self.CIE_10['CODIGO_AGRUP']
        self.desc_agrup_list = self.CIE_10['DESCRIPCION_AGRUP']

        self.w2v_model = gensim.models.Word2Vec.load(config.W2V_IR)

        self.dictionary = gensim.corpora.Dictionary.load(config.DICTIONARY)
        self.lda = gensim.models.ldamodel.LdaModel.load(config.LDA)
        self.index = gensim.similarities.MatrixSimilarity.load(config.INDEX)

        self.model_IR = load_model(config.MODELO_RANK)

        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(module)s %(funcName)s line:%(lineno)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(os.path.join(config.LOGS_PATH, 'classifier_%s_DEBUG.log'
                                                   % datetime.now().strftime('%Y%m%d')))
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.DEBUG)
        self.logger.handlers = []
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        self.logger.debug('Classifier process called.')
        self.cie_languages = {}
        self.cie_languages['I00-I02'] = {'esp': 'Fiebre reumática aguda', 'cat': 'Febre reumàtica aguda'}
        self.cie_languages['I05-I09'] = {'esp': 'Enfermedades reumáticas crónicas cardiacas',
                                   'cat': 'Cardiopaties reumàtiques cròniques'}
        self.cie_languages['I10-I16'] = {'esp': 'Enfermedades hipertensivas', 'cat': 'Malalties hipertensives'}
        self.cie_languages['I20-I25'] = {'esp': 'Enfermedades isquémicas cardíacas', 'cat': 'Cardiopaties isquèmiques'}
        self.cie_languages['I26-I28'] = {'esp': 'Enfermedad cardiaca pulmonar y enfermedades de la circulación pulmonar',
                                   'cat': 'Malaltia cardiopulmona i malalties de la ciculació pulmonar'}
        self.cie_languages['I30-I52'] = {'esp': 'Otras formas de enfermedad cardiaca', 'cat': 'Altres formes de cardiopatia'}
        self.cie_languages['I60-I69'] = {'esp': 'Enfermedades cerebrovasculares','cat': 'Malalties cerebrovasculars'}
        self.cie_languages['I70-I79'] = {'esp': 'Enfermedades de arterias, arteriolas y capilares',
                                   'cat': "Malalties d'artèries, arterioles i capil·lars"}
        self.cie_languages['I80-I89'] = {'esp': 'Enfermedades de vena, vasos linfáticos y nodos linfáticos, no clasificados en otra parte',
                                   'cat': 'Malalties de venes, vasos limfàtics i ganglis limfàtics no classificades a cap altre lloc'}
        self.cie_languages['I95-I99'] = {'esp': 'Otros trastornos del sistema circulatorio y trastornos sin especificar',
                                   'cat': "Altres trastorns de l'aparell circulatori i trastorns de l'aparell circularoti no especificats"}


    ################################################################################
    ############# FUNCIONES Y EXPLOTACIÓN DEL MODELO DE PROBABILIDADES #############
    ################################################################################

    def delete_accents_and_lower(self, text, **kwargs):
        if 'encoding' in kwargs:
            unicode_text = text.decode(kwargs['encoding'])
        else:
            unicode_text = text
        nkfd_form = unicodedata.normalize('NFKD', unicode_text)
        no_accents = u"".join([char for char in nkfd_form if not unicodedata.combining(char)])
        lower_no_accents = no_accents.lower()
        lower_no_accents_no_enie = lower_no_accents.replace('ñ'.decode('utf-8'), 'n')
        if 'encoding' in kwargs:
            return lower_no_accents_no_enie
        else:
            return lower_no_accents_no_enie

    def text_preparation(self, text):
        lower_no_accents = self.delete_accents_and_lower(text)
        strange_characters = re.sub('[^a-zA-Zñ<> ]', '', lower_no_accents)
        extra_white_spaces = " ".join(strange_characters.split())
        return extra_white_spaces.strip()

    def create_dictionary_from_csv(self, dict_path):
        try:
            readfile = open(os.path.join(config.DICTS, dict_path), 'r')
        except:
            raise Exception('The %s dictionary is not in the right path' % dict_path)
        data = {}
        for line in readfile:
            fields = line.replace('\n', '').split(';')
            error = fields[0]
            correction = fields[1]
            data[error] = correction
        return data


    def replace_content(cls, text, dict_replace, **kwargs):
        if 'encoding' in kwargs:
            unicode_text = text.decode(kwargs['encoding'])
        else:
            unicode_text = text
        clean_text = unicode_text
        for key in dict_replace.keys():
            try:
                clean_text = clean_text.replace(' %s ' % key, ' %s ' % dict_replace[key])
            except:
                print dict_replace[key]
                print clean_text
            if clean_text.startswith('%s ' % key):
                clean_text = clean_text.replace('%s ' % key, '%s ' % dict_replace[key])
            if clean_text.endswith(' %s' % key):
                clean_text = clean_text.replace(' %s' % key, ' %s' % dict_replace[key])
        if 'encoding' in kwargs:
            return clean_text.encode(kwargs['encoding'])
        else:
            return clean_text

    def apply_dict(self,dat, diccionario):
        pattern = re.compile(r'\b(' + '|'.join(diccionario.keys()) + r')\b')
        tmp1 = dat
        dat = [pattern.sub(lambda x: diccionario[x.group()], item) for item in tmp1]
        return dat

    #def explotacion_LSTM(self, text):
    #    try:
    #        self.logger.debug('The following text is going to be processed: %s' % text)
    #        corr_dict = self.create_dictionary_from_csv('PD_01_Diccionario_correc.csv')
    #        lex_dict = self.create_dictionary_from_csv('PD_02_Diccionario_lex.csv')
#
    #        prepared_text = self.replace_content(text, corr_dict)
    #        prepared_text = self.replace_content(prepared_text, lex_dict)
#
    #        df_prepared_text = pd.DataFrame([prepared_text])
    #        df_prepared_text.columns = ['DOC_CONTENT']
    #        corrected_text = np.array(df_prepared_text['DOC_CONTENT'].tolist())
    #        array_textos = [corrected_text[0].split(" ")]
    #        array_textos[0] = array_textos[0][0:200]
#
    #        L = [x.replace('\r','') for x in array_textos[0]]
    #        self.logger.debug('The following text is going to be classified: %s' % ' '.join(L))
#
    #        pkl_file = open(os.path.join(config.MODEL, 'word2idx.pkl'), 'rb')
    #        word2idx = pickle.load(pkl_file)
    #        a = [0] * self.embedding_size
    #        for j in range(0, len(array_textos[0])):
    #            if array_textos[0][j] in word2idx.keys():
    #                a[j] = word2idx[array_textos[0][j]]
    #        X_tokens = np.array([a])
#
    #        dic_cat = pd.read_csv(os.path.join(config.DICTS, 'CD_01_Diccionario_categorias.csv'),
    #                              sep=";", encoding='utf-8')
#
    #        with self.graph.as_default():
    #            score = self.model_LSTM.predict(X_tokens)
    #        dic_cat['SCORE'] = score[0]
    #        dic_cat = dic_cat.sort_values('SCORE', ascending = False)
    #        main_result = []
    #        if (dic_cat.iloc[0]['SCORE'] - dic_cat.iloc[1]['SCORE']) >= 0.30:
    #            result = {}
    #            result['codigo_cie'] = dic_cat.iloc[0]['CODIGO_CIE']
    #            result['nombre'] = dic_cat.iloc[0]['NOMBRE']
    #            result['score'] = str(dic_cat.iloc[0]['SCORE'])
    #            main_result.append(result)
    #        elif (dic_cat.iloc[0]['SCORE'] - dic_cat.iloc[1]['SCORE']) < 0.30 and \
    #                (dic_cat.iloc[1]['SCORE'] - dic_cat.iloc[2]['SCORE']) < 0.30:
    #            result1 = {}
    #            result1['codigo_cie'] = dic_cat.iloc[0]['CODIGO_CIE']
    #            result1['nombre'] = dic_cat.iloc[0]['NOMBRE']
    #            result1['score'] = str(dic_cat.iloc[0]['SCORE'])
    #            result2 = {}
    #            result2['codigo_cie'] = dic_cat.iloc[1]['CODIGO_CIE']
    #            result2['nombre'] = dic_cat.iloc[1]['NOMBRE']
    #            result2['score'] = str(dic_cat.iloc[1]['SCORE'])
    #            result3 = {}
    #            result3['codigo_cie'] = dic_cat.iloc[2]['CODIGO_CIE']
    #            result3['nombre'] = dic_cat.iloc[2]['NOMBRE']
    #            result3['score'] = str(dic_cat.iloc[2]['SCORE'])
    #            main_result.append(result1)
    #            main_result.append(result2)
    #            main_result.append(result3)
    #        elif (dic_cat.iloc[0]['SCORE'] - dic_cat.iloc[1]['SCORE']) < 0.30 and \
    #                (dic_cat.iloc[1]['SCORE'] - dic_cat.iloc[2]['SCORE']) >= 0.30:
    #            result1 = {}
    #            result1['codigo_cie'] = dic_cat.iloc[0]['CODIGO_CIE']
    #            result1['nombre'] = dic_cat.iloc[0]['NOMBRE']
    #            result1['score'] = str(dic_cat.iloc[0]['SCORE'])
    #            result2 = {}
    #            result2['codigo_cie'] = dic_cat.iloc[1]['CODIGO_CIE']
    #            result2['nombre'] = dic_cat.iloc[1]['NOMBRE']
    #            result2['score'] = str(dic_cat.iloc[1]['SCORE'])
    #            main_result.append(result1)
    #            main_result.append(result2)
    #        else:
    #            result = {}
    #            result['error'] = ['No se ha podido clasificar el texto introducido']
    #            main_result.append(result)
    #        return main_result
    #    except Exception as e:
    #        return e.message, type(e)

    def explotacion_LSTM(self, text, language):
        try:
            self.logger.debug('The following text is going to be processed: %s' % text)
            corr_dict = self.create_dictionary_from_csv('PD_01_Diccionario_correc.csv')
            lex_dict = self.create_dictionary_from_csv('PD_02_Diccionario_lex.csv')

            prepared_text = self.replace_content(text, corr_dict)
            prepared_text = self.replace_content(prepared_text, lex_dict)

            df_prepared_text = pd.DataFrame([prepared_text])
            df_prepared_text.columns = ['DOC_CONTENT']
            corrected_text = np.array(df_prepared_text['DOC_CONTENT'].tolist())
            array_textos = [corrected_text[0].split(" ")]
            array_textos[0] = array_textos[0][0:200]

            L = [x.replace('\r','') for x in array_textos[0]]
            self.logger.debug('The following text is going to be classified: %s' % ' '.join(L))

            pkl_file = open(os.path.join(config.MODEL, 'word2idx.pkl'), 'rb')
            word2idx = pickle.load(pkl_file)
            a = [0] * self.embedding_size
            for j in range(0, len(array_textos[0])):
                if array_textos[0][j] in word2idx.keys():
                    a[j] = word2idx[array_textos[0][j]]
            X_tokens = np.array([a])

            dic_cat = pd.read_csv(os.path.join(config.DICTS, 'CD_01_Diccionario_categorias.csv'),
                                  sep=";", encoding='utf-8')

            with self.graph.as_default():
                score = self.model_LSTM.predict(X_tokens)
            dic_cat['SCORE'] = score[0]
            dic_cat = dic_cat.sort_values('SCORE', ascending = False)
            main_result = []
            for e in range(3):
                result = {}
                result['codigo_cie'] = dic_cat.iloc[e]['CODIGO_CIE']
                result['nombre'] = self.cie_languages[result['codigo_cie']][language]
                score = str(round(dic_cat.iloc[e]['SCORE'] * 100, 1))
                if round(dic_cat.iloc[e]['SCORE'] * 100, 1) < 0.1:
                   score = 'Menor que 0.1'
                if score.endswith('.0'):
                    score = score[:-2]
                result['score'] = score
                main_result.append(result)
            return main_result
        except Exception as e:
            print e.message
            print type(e)
            result = {}
            result['error'] = 'No se ha podido clasificar el texto introducido'
            return result

    #################################################################################
    ################ FUNCIONES Y EXPLOTACIÓN DEL MODELO DE DISTANCIA ################
    #################################################################################

    def correcciones(self, desc_list):
        df_desc_list = pd.DataFrame(desc_list)

        df_desc_list = df_desc_list.dropna()

        df_desc_list["DOC_CONTENT2"] = df_desc_list["DESCRIPCION"].str.lower()
        df_desc_list["DOC_CONTENT3"] = df_desc_list["DOC_CONTENT2"].apply(
            lambda x: re.sub(r'[\'\"]', r'', x.encode('UTF-8')))
        df_desc_list["DOC_CONTENT4"] = df_desc_list["DOC_CONTENT3"].str.replace(r"\b(\\xa0)\b", r"")
        df_desc_list["DOC_CONTENT5"] = df_desc_list["DOC_CONTENT4"].str.replace(r"\b(\\n)\b", r"")
        df_desc_list["DOC_CONTENT6"] = df_desc_list["DOC_CONTENT5"].apply(
            lambda x: re.sub(r'[^a-zA-ZÁáÉéÍíÓóÚúÜüÑñ\s]', r' ', x))
        df_desc_list["DOC_CONTENT7"] = df_desc_list["DOC_CONTENT6"].apply(lambda x: re.sub(r'\s+', r' ', x)).str.strip()

        #Quitamos los vacíos de nuevo
        df_desc_list = df_desc_list.loc[df_desc_list["DOC_CONTENT7"] != ""]

        #Corpus final
        df_desc_list = df_desc_list[["DOC_CONTENT7"]]
        df_desc_list.columns = ["DESCRIPCION"]

        dic_lex = \
            pd.read_csv(config.DIC_LEX_IR, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
                orient='dict')['CORRECCION']
        df_desc_list["DESCRIPCION"] = self.apply_dict(df_desc_list["DESCRIPCION"], dic_lex)

        dic_corr = \
            pd.read_csv(config.DIC_CORR_2, sep=";", index_col=0, converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
                orient='dict')['CORRECCION']
        df_desc_list["DESCRIPCION"] = self.apply_dict(df_desc_list["DESCRIPCION"], dic_corr)

        dic_ngramas = \
            pd.read_csv(config.DICT_NGRAMMA, sep=";", index_col=0,
                        converters={'TEXTO': str, 'CORRECCION': str}).to_dict(
                orient='dict')['CORRECCION']
        df_desc_list["DESCRIPCION"] = self.apply_dict(df_desc_list["DESCRIPCION"], dic_ngramas)

        desc_list_corrected = df_desc_list['DESCRIPCION']

        return desc_list_corrected

    def evaluar_IR_dist(self, sentence):
        #RELEVANCIA TFIDF
        response = self.tfv.transform([sentence])

        tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

        terminos = []
        for col in response.nonzero()[1]:
            terminos.append([self.feature_names[col], response[0, col]])

        terminos = pd.DataFrame(terminos)
        terminos.columns = ['TERMINO', 'VALOR']

        ## Fijamos el peso de las palabras no existentes en la tf-idf en el percentil 5
        terminos_tfidf = pd.concat([pd.DataFrame(self.feature_names), pd.DataFrame(self.tfv.idf_)], axis=1)
        terminos_tfidf.columns = ['TERMINO', 'VALOR']
        terminos_tfidf.sort_values(by=['VALOR'], ascending=False)

        peso_palabras_nuevas = np.percentile(terminos_tfidf['VALOR'], 0.01, axis=0)
        valor_0 = terminos['VALOR'][0]
        termino_0 = terminos_tfidf[terminos_tfidf['TERMINO'] == terminos['TERMINO'][0]].reset_index()['VALOR'][0]
        peso_nueva_normalizado = (peso_palabras_nuevas * valor_0) / termino_0

        words = sentence.split()

        for word in words:
            if word.decode('utf-8') not in terminos['TERMINO'].tolist():
                terminos = terminos.append({'TERMINO': word, 'VALOR': peso_nueva_normalizado}, ignore_index=True)

        terminos = terminos.sort_values(by=['VALOR'], ascending=False)

        #Búsqueda:
        term = [termino for termino in terminos['TERMINO']]
        palabras = [descripcion.split(" ") for descripcion in self.desc_list]
        dist_desc = [self.seleccion_entidad(palabra, termino) for palabra in palabras for termino in term]

        df_distancias = pd.DataFrame(np.asarray(pd.DataFrame(dist_desc)[0]).reshape((len(self.CIE_10), len(terminos))))

        epsilon = 0.001
        distancias = [distancia[0] for distancia in dist_desc]
        eps_dist = [(df_distancias[indice] + epsilon) / terminos['VALOR'][indice] for indice in range(0, len(terminos))]

        df_distancias_eps = pd.DataFrame(np.asarray(pd.DataFrame(eps_dist)).T)

        ## Suma columnas
        suma = pd.DataFrame(df_distancias_eps.sum(axis=1))
        suma.columns = ['SUMA']
        df_distancias_eps = pd.concat([df_distancias_eps, suma], axis=1)

        categoria = self.locate_min(suma['SUMA'], self.cod_list)[1][0]

        return categoria, suma

    def evaluar_IR_w2v(self, sentence):
        #RELEVANCIA TFIDF
        response = self.tfv.transform([sentence])

        tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

        terminos = []
        for col in response.nonzero()[1]:
            terminos.append([self.feature_names[col], response[0, col]])

        terminos = pd.DataFrame(terminos)
        terminos.columns = ['TERMINO', 'VALOR']

        #Fijamos el peso de las palabras no existentes en la tf-idf en el percentil 5
        terminos_tfidf = pd.concat([pd.DataFrame(self.feature_names), pd.DataFrame(self.tfv.idf_)], axis=1)
        terminos_tfidf.columns = ['TERMINO', 'VALOR']
        terminos_tfidf.sort_values(by=['VALOR'], ascending=False)

        peso_palabras_nuevas = np.percentile(terminos_tfidf['VALOR'], 0.01, axis=0)

        valor_0 = terminos['VALOR'][0]
        termino_0 = terminos_tfidf[terminos_tfidf['TERMINO'] == terminos['TERMINO'][0]].reset_index()['VALOR'][0]
        peso_nueva_normalizado = (peso_palabras_nuevas * valor_0) / termino_0

        words = sentence.split()

        for word in words:
            if word not in terminos['TERMINO'].tolist():
                terminos = terminos.append({'TERMINO': word, 'VALOR': peso_nueva_normalizado}, ignore_index=True)

        terminos = terminos.sort_values(by=['VALOR'], ascending=False)

        #Fijamos número de términos relevantes
        numero_terminos = min(len(sentence.split(' ')), 5)  #Evaluar número óptimo de términos relevantes
        terminos_relavantes = terminos.head(numero_terminos)
        terminos_relavantes = terminos_relavantes.reset_index()

        #TOP3 DISTANCIA COSENO W2V

        #Seleccionamos el top3 de distancias de coseno min de w2v:
        top3_dist_w2v = []
        for word in terminos_relavantes.TERMINO:
            palabras = []
            if word in self.w2v_model.wv.vocab:
                related = self.w2v_model.wv.most_similar_cosmul(positive=word)
                related = pd.DataFrame(related).head(3)  #Evaluar si coger top n o distancia menor que x
                for term in related[0]:
                    palabras += [term]

            top3_dist_w2v = top3_dist_w2v + [palabras]
        p95 = 9.72548
        dist_w2v = pd.DataFrame(self.cod_list)
        for sinonimos in top3_dist_w2v:
            #Si el término existe en w2v asociamos el vector de mínimos
            if len(sinonimos) > 0:
                term = [sinonimo for sinonimo in sinonimos]
                palabras = [descripcion.split(" ") for descripcion in self.desc_list]
                dist_desc = [self.seleccion_entidad(palabra, termino) for palabra in palabras for termino in term]

                df_distancias = pd.DataFrame(
                    np.asarray(pd.DataFrame(dist_desc)[0]).reshape((len(self.CIE_10), len(sinonimos))))
                minimo_fila = pd.DataFrame(df_distancias.min(axis=1))
                minimo_fila.columns = ['MIN']
                dist_w2v = pd.concat([dist_w2v, minimo_fila], axis=1)
            #Si el término no está en el w2v asociamos el p95 al término
            else:
                dist_p95 = pd.DataFrame(p95, index=range(len(self.CIE_10)), columns=range(1))
                dist_p95.columns = ['dist_p95']
                dist_w2v = pd.concat([dist_w2v, dist_p95], axis=1)

        del dist_w2v['CODIGO']
        dist_w2v.columns = list(range(len(top3_dist_w2v)))

        #Ponderamos por la relevancia de los términos en la tf-idf
        epsilon = 0.001
        #distancias = [distancia[0] for distancia in dist_desc]
        eps_dist = [(dist_w2v[indice] + epsilon) / terminos_relavantes['VALOR'][indice] for indice in
                    range(0, len(terminos_relavantes))]
        df_distancias_eps = pd.DataFrame(np.asarray(pd.DataFrame(eps_dist)).T)

        #Hallamos la distancia mínima como la suma de las columnas
        suma = pd.DataFrame(df_distancias_eps.sum(axis=1))
        suma.columns = ['SUMA']
        df_distancias_eps = pd.concat([df_distancias_eps, suma], axis=1)

        categoria = self.locate_min(suma['SUMA'], self.cod_list)[1][0]

        return categoria, suma

    def damerau_levenshtein_distance(self, s1, s2):
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

    def seleccion_entidad(self, entidad_list, ent_input):
        dist = []
        for ent in entidad_list:
            dist.append(self.damerau_levenshtein_distance(ent_input, ent))
        entidad = self.locate_min(dist, entidad_list)

        return entidad

    def locate_min3(self, df, entidad_list):
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

    def locate_min(self, a, entidad_list):
        smallest = min(np.float64(a))
        indices = [entidad_list[index] for index, element in enumerate(np.float64(a)) if
                   np.float64(smallest) == np.float64(element)]

        return smallest, indices

    def locate_max_float(self, a, entidad_list):
        smallest = max(np.float64(a))
        indices = [entidad_list[index] for index, element in enumerate(np.float64(a)) if
                   np.float64(smallest) == np.float64(element)]
        return smallest, indices

    def evaluar_IR_LDA(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub('á', 'a', re.sub('é', 'e', re.sub('í', 'i', re.sub('ó', 'o', re.sub('ú', 'u', sentence)))))
        test = [sentence]
        test_texts = [[token for token in text.split(' ')] for text in test]
        test_corpus = [self.dictionary.doc2bow(text) for text in test_texts]

        vec_lda = self.lda[test_corpus]
        sims = self.index[vec_lda]
        df_sims = pd.DataFrame(sims[0])

        categoria = self.locate_max_float(df_sims, self.cod_list)[1][0]
        return categoria, df_sims

    def explotacion_IR(self, original_sentence, language):
        try:
            original_sentence = original_sentence.lower()
            original_sentence = re.sub('à', 'a', re.sub('è', 'e', re.sub('ì', 'i', re.sub('ò', 'o', re.sub('ù', 'u',
                                                                                                           original_sentence)))))
            #Explotamos modelos
            sentence = original_sentence.decode('utf-8')
            sentence = pd.DataFrame([sentence], columns=['DESCRIPCION'])
            sentence = self.correcciones(sentence)[0]
            #Llamada IR distancias
            resultado_dist, vect_distancias = self.evaluar_IR_dist(sentence)

            #Llamada IR w2v
            resultado_w2v, vect_w2v = self.evaluar_IR_w2v(sentence)

            ## Llamada IR LSI
            resultado_LDA, vect_LDA = self.evaluar_IR_LDA(original_sentence)
            vectores = [vect_distancias, vect_w2v, vect_LDA]

            #Cálculo score
            with self.graph.as_default():
                score = self.model_IR.predict(
                [np.asarray(vect_distancias.transpose()), np.asarray(vect_w2v.transpose()),
                 np.asarray(vect_LDA.transpose())])

            categorias = [resultado_dist, resultado_w2v, resultado_LDA]
            resultado = categorias[np.argmax(score[0])]

            df = pd.concat([self.cod_agrup_list, self.desc_agrup_list, vectores[np.argmax(score[0])]], axis=1)
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
            main_result = []
            for e in range(5):
                result1 = {}
                result1['codigo_cie'] = result.iloc[e]['CODIGO_AGRUP']
                result1['nombre'] = self.cie_languages[result1['codigo_cie']][language]
                distancia = str(round(result.iloc[e]['DIST'], 1))
                if distancia.endswith('.0'):
                    distancia = distancia[:-2]
                result1['distancia'] = distancia
                main_result.append(result1)
            return main_result
        except:
            result = {}
            result['error'] = 'No se ha podido clasificar el texto introducido por distancia'
            return result

    def explotacion_conjunta(self, texto, language):
        modelo_1 = self.explotacion_LSTM(texto, language)
        modelo_2 = self.explotacion_IR(texto, language)
        result = {}
        #result['Modelo1'] = modelo_1
        #result['Modelo2'] = modelo_2

        if not isinstance(modelo_1, list):
            if modelo_1.has_key('error'):
                result['error'] = modelo_1['error']
            else:
                result['Modelo1'] = modelo_1
        else:
            result['Modelo1'] = modelo_1

        if not isinstance(modelo_2, list):
            if modelo_2.has_key('error'):
                result['Error'] = modelo_2['error']
            else:
                result['Modelo2'] = modelo_2
        else:
            result['Modelo2'] = modelo_2

        return result