# api/urls.py (append)
from django.urls import path
from .views import ingest_view

urlpatterns = [
    path('ingest/', ingest_view, name='ingest'),
]
