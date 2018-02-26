from django.conf.urls import url, include
from django.contrib import admin
from . import views

app_name = 'CodeMyHealth'
urlpatterns = [
	url(r'^home', views.home, name='home'),
	url(r'^anexo_diagnostico', views.anexo_diagnostico, name='anexo_diagnostico'),
	url(r'^getDiagnostico', views.getDiagnostico, name='getDiagnostico'),
	url(r'^obtain_text_from_pdf_file', views.obtain_text_from_pdf_file, name='obtain_text_from_pdf_file'),
	url(r'^obtain_text_from_docx_file', views.obtain_text_from_docx_file, name='obtain_text_from_docx_file'),
	url(r'^obtain_text_from_text_file', views.obtain_text_from_text_file, name='obtain_text_from_text_file'),
]
