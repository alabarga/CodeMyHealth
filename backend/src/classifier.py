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


class Classifier(object):
    def __init__(self):
        self.embedding_size = 200
        self.model = load_model(os.path.join(config.MODEL, 'BI_LSTM_50_lex_corr.h5'))
        self.graph = tf.get_default_graph()

        self.logger = logging.getLogger('main')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s [%(module)s %(funcName)s line:%(lineno)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(os.path.join(config.LOGS_PATH, 'harmonization_%s_DEBUG.log'
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
        pdb.set_trace()
        tmp1 = dat
        dat = [pattern.sub(lambda x: diccionario[x.group()], item) for item in tmp1]
        return dat

    def explotacion(self, text):
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
                score = self.model.predict(X_tokens)

            results={'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
            dic_cat['SCORE'] = score[0]
            for i in range(0,len(results)):
                results[str(i)].append(dic_cat['CODIGO_CIE'][i])
                results[str(i)].append(dic_cat['NOMBRE'][i])
                results[str(i)].append(str(dic_cat['SCORE'][i]))

            return results
        except Exception as e:
            return e.message, type(e)
