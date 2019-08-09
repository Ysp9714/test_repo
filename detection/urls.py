from django.urls import path

from .views import *

urlpatterns = [
    path('result/<int:camera_type>/', DetectionResult, name='DetectionResult'),
    path('')
]
