from rest_framework import serializers
from .models import Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'image', 'result', 'accuracy', 'created_at', 'crop_class']
        read_only_fields = ['result', 'accuracy', 'crop_class']