from django import forms
from .models import Prediction
from .model_loader import get_predictor

class ImageUploadForm(forms.ModelForm):
    # Add a crop class selection field
    CROP_CHOICES = [
        ('maize', 'Maize'),
        ('onion', 'Onion'),
        ('tomato', 'Tomato')
    ]
    
    crop_class = forms.ChoiceField(
        choices=CROP_CHOICES,
        required=True,
        label='Select Plant Type',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = Prediction
        fields = ['image', 'crop_class']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control'})
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Make the crop selection field required
        self.fields['crop_class'].help_text = "Please select the plant type to improve prediction accuracy"