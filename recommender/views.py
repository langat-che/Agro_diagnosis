from django.shortcuts import render, redirect
from django.conf import settings
from .model_loader import get_predictor
from .forms import ImageUploadForm
from .models import Prediction
import os
from rest_framework import viewsets
from .serializers import PredictionSerializer

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    def perform_create(self, serializer):
        instance = serializer.save()
        prediction_result, accuracy, crop_class = Prediction(instance.image.path)
        instance.result = prediction_result
        instance.accuracy = accuracy
        instance.crop_class = crop_class
        instance.save()

def home(request):
    """Home page with upload form"""
    # Get crop options for the dropdown
    predictor = get_predictor()
    crop_options = predictor.get_crop_options()
    
    form = ImageUploadForm()
    return render(request, 'home.html', {'form': form, 'crop_options': crop_options})


def predict_image(request):
    """Handle image upload and make prediction"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            image = form.save(commit=False)
            image.crop_class = form.cleaned_data['crop_class']
            image.save()
            
            # Get the image path
            image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))
            
            # Get predictor and make prediction with the selected crop
            predictor = get_predictor()
            result = predictor.predict(image_path, selected_crop=image.crop_class)
            
            # Update the prediction object with results
            image.result = result['disease']
            image.accuracy = result['confidence']
            image.save()
            
            # Check if we have subclasses to choose from
            possible_subclasses = result.get('possible_subclasses', [])
            
            # Pass the prediction results to the template
            context = {
                'image_url': image.image.url,
                'disease': result['disease'],
                'subclass': result.get('subclass', ''),  
                'confidence': result['confidence'],
                'cause': result.get('cause', ''),
                'remedy': result.get('remedy', ''),
                'remedy_organic': result.get("remedy_organic", ""),
                'remedy_chemical': result.get("remedy_chemical", ""),
                'prediction_id': image.id,
                'crop_class': image.crop_class,
                'possible_subclasses': possible_subclasses
            }
            
            return render(request, 'results.html', context)
    else:
        # Get crop options for the dropdown
        predictor = get_predictor()
        crop_options = predictor.get_crop_options()
        
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form, 'crop_options': crop_options})


def select_subclass(request, prediction_id):
    """Handle selection of a specific subclass after initial prediction"""
    if request.method == 'POST':
        selected_subclass = request.POST.get('subclass')
        
        # Get the prediction object
        try:
            prediction = Prediction.objects.get(id=prediction_id)
            image_path = os.path.join(settings.MEDIA_ROOT, str(prediction.image))
            
            # Get new prediction with selected subclass
            predictor = get_predictor()
            result = predictor.predict_with_subclass(
                image_path, 
                selected_crop=prediction.crop_class,
                selected_subclass=selected_subclass
            )
            
            # Update the prediction object
            prediction.result = result['disease']
            prediction.accuracy = result['confidence']
            prediction.save()
            
            # Pass the updated results to the template
            context = {
                'image_url': prediction.image.url,
                'disease': result['disease'],
                'subclass': result.get('subclass', ''),
                'confidence': result['confidence'],
                'cause': result.get('cause', ''),
                'remedy': result.get('remedy', ''),
                'remedy_organic': result.get("remedy_organic", ""),
                'remedy_chemical': result.get("remedy_chemical", ""),
                'prediction_id': prediction.id,
                'crop_class': prediction.crop_class,
                'possible_subclasses': result.get('possible_subclasses', [])
            }
            
            return render(request, 'results.html', context)
            
        except Prediction.DoesNotExist:
            return redirect('home')
            
    return redirect('home')


def about(request):
    return render(request, 'about.html')