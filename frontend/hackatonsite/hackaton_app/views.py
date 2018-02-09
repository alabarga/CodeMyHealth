# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .forms import UploadFileForm

##Modulo de importacion de PDF a text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO

##Modulo importacion de DOCX a text
from docx import Document

##Modulo importacion de DOC a text
import fulltext


import requests
import json
import sys
import logging
import os
#sys.path.append('/home/ubuntu/hackaton/hackatonsite/backend/src/')

#from classifier import Classifier

# Create your views here.

########################
###### HOME VIEW #######
########################
def home(request):
    return render(request, 'redactar_diagnostico.html')

def anexo_diagnostico(request):
    return render(request, 'anexo_diagnostico.html')

@csrf_exempt
def getDiagnostico(request):
	data_unicode = request.body
	data = json.loads(data_unicode)
	result_set = {"response":[],"error":"" }
	try:
		##llamada al servicio flask que me va a devolver los pesos de los diferentes cies
		url = 'http://%s:%s/predict' % ('localhost', '5000')
		json_data = json.dumps({'texto':data['CIE']}).encode('utf-8')
		headers = {'content-type': 'application/json'}
		r = requests.post(url, data=json_data, headers=headers)
		
		if r.status_code == 200:
			result_set['response'] = json.loads(r.content)
		else:
			print "hola"
			print r.content
			result_set['error'] = "El servicio REST ha fallado, intentelo más tarde."
			
	except Exception as e:
		print '%s (%s)' % (e.message, type(e))
		result_set["error"] = "No se ha podido realizar la peticion REST al servicio, intentelo más tarde."
	return JsonResponse(result_set, safe=False)

@csrf_exempt
def obtain_text_from_pdf_file(request):
	result_set = {"response":[],"error":"" }
	print request.FILES['file']
	try:
		## Uso de la libreria PDFminer (pdf2text)
		texto = convert_pdf_to_txt(request.FILES['file'])
		if not texto:
			result_set["error"] = "Algo fallo al convertir el PDF a texto"
			return JsonResponse(result_set, safe=False)
		##llamada al servicio flask que me va a devolver los pesos de los diferentes cies
		url = 'http://%s:%s/predict' % ('localhost', '5000')
		json_data = json.dumps({'texto': texto}).encode('utf-8')
		headers = {'content-type': 'application/json'}
		r = requests.post(url, data=json_data, headers=headers)
		
		if r.status_code == 200:
			result_set['response'] = json.loads(r.content)
		else:
			result_set['error'] = "El servicio REST ha fallado, intentelo más tarde."
			
	except Exception as e:
		print '%s (%s)' % (e.message, type(e))
		result_set["error"] = "No se ha podido realizar la peticion REST al servicio, intentelo más tarde."
	return JsonResponse(result_set, safe=False)

@csrf_exempt
def obtain_text_from_docx_file(request):
	result_set = {"response":[],"error":"" }
	try:
		## Uso de la libreria PDFminer (pdf2text)
		texto = convert_docx_to_txt(request.FILES['file'])
		if not texto:
			result_set["error"] = "Algo fallo al convertir el DOCX a texto"
			return JsonResponse(result_set, safe=False)
		##llamada al servicio flask que me va a devolver los pesos de los diferentes cies
		url = 'http://%s:%s/predict' % ('localhost', '5000')
		json_data = json.dumps({'texto': texto}).encode('utf-8')
		headers = {'content-type': 'application/json'}
		r = requests.post(url, data=json_data, headers=headers)
		
		if r.status_code == 200:
			result_set['response'] = json.loads(r.content)
		else:
			result_set['error'] = "El servicio REST ha fallado, intentelo más tarde."
			
	except Exception as e:
		print '%s (%s)' % (e.message, type(e))
		result_set["error"] = "No se ha podido realizar la peticion REST al servicio, intentelo más tarde."
	return JsonResponse(result_set, safe=False)

@csrf_exempt
def obtain_text_from_doc_file(request):
	result_set = {"response":[],"error":"" }
	try:
		## Uso de la libreria fulltext
		print "llego"
		print request.FILES['file']
		texto = convert_doc_to_txt(request.FILES['file'])
		print "me voy"
		if not texto:
			result_set["error"] = "Algo fallo al convertir el DOCX a texto"
			return JsonResponse(result_set, safe=False)
		##llamada al servicio flask que me va a devolver los pesos de los diferentes cies
		url = 'http://%s:%s/predict' % ('localhost', '5000')
		json_data = json.dumps({'texto': texto}).encode('utf-8')
		headers = {'content-type': 'application/json'}
		r = requests.post(url, data=json_data, headers=headers)
		
		if r.status_code == 200:
			result_set['response'] = json.loads(r.content)
		else:
			result_set['error'] = "El servicio REST ha fallado, intentelo más tarde."
			
	except Exception as e:
		print '%s (%s)' % (e.message, type(e))
		result_set["error"] = "No se ha podido realizar la peticion REST al servicio, intentelo más tarde."
	return JsonResponse(result_set, safe=False)

##########################
## Funciones auxiliares ##
##########################
def convert_pdf_to_txt(file):
	try:
		rsrcmgr = PDFResourceManager()
		retstr = StringIO()
		codec = 'utf-8'
		laparams = LAParams()
		device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
		interpreter = PDFPageInterpreter(rsrcmgr, device)
		password = ""
		maxpages = 0
		caching = True
		pagenos=set()
		for page in PDFPage.get_pages(file, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
			interpreter.process_page(page)
		file.close()
		device.close()
		str = retstr.getvalue()
		retstr.close()
		return str
	except Exception as e:
		print '%s (%s)' % (e.message, type(e))
		return ""

def convert_docx_to_txt(file):
	document = Document(file)
	fullText = []
	for p in document.paragraphs:
		fullText.append(p.text)
	print fullText
	return '\n'.join(fullText)

def convert_doc_to_txt(file):
	fulltext.get(file)