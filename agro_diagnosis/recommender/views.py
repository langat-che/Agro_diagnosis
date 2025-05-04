from django.shortcuts import render, redirect
from django.conf import settings
from .model_loader import get_predictor
from .forms import ImageUploadForm
from .models import Prediction
import os
import time


def home(request):
    """Home page with upload form"""
    form = ImageUploadForm()
    return render(request, 'home.html', {'form': form})

def predict_image(request):
    """Handle image upload and make prediction"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image = form.save(commit=False)
            image.save()
            
            # Get the image path
            image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))
            
            # Get predictor and make prediction
            predictor = get_predictor()
            result = predictor.predict(image_path)
            
            # Update the prediction object with results
            image.result = result['disease']
            image.accuracy = result['confidence']
            image.save()
            
            # Pass the prediction results to the template
            context = {
                'image_url': image.image.url,
                'disease': result['disease'],
                'subclass': result['subclass'], 
                'confidence': result['confidence'],
                'cause': result['cause'],
                'remedy': result['remedy'],
                'healthy_image': result['healthy_image'],
                'prediction_id': image.id
            }
            
            return render(request, 'results.html', context)
    else:
        form = ImageUploadForm()
    
    return render(request, 'home.html', {'form': form})

def about(request):
    return render(request, 'about.html')