from django.urls import path
from . import views
from rest_framework.routers import DefaultRouter
from .views import PredictionViewSet

router = DefaultRouter()
router.register(r'predictions', PredictionViewSet)

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_image, name='predict_image'),
    path('about/', views.about, name='about'),
    path('select-subclass/<int:prediction_id>/', views.select_subclass, name='select_subclass'),
]