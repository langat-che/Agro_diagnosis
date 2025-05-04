from django import forms
from .models import Prediction

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Prediction
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
        }