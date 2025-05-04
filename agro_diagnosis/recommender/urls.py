from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_image, name='predict_image'),
    path('about/', views.about, name='about'),
]