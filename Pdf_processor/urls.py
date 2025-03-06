from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('home/', views.show_home, name='show_home'),
    
    path('pdfs/', views.show_pdf_list, name='show_pdf_json'),
    path('pdfs/<str:pdf_filename>.json', views.download_json, name='download_json'),
    path('delete_document/<str:document_id>/', views.delete_document, name='delete_document'),
    path('chatbot', views.chatbot_view, name='chatbot'),




]
