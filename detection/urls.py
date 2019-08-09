from django.urls import path

from .views import *

urlpatterns = [
    path('action_result/<int:camera_type>/', ActionResult, name='ActionResult'),
    path('detection_result/<int:camera_type>/', DetectionResult, name='DetectionResult')
]