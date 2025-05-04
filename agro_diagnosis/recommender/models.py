from django.db import models


class Prediction(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result = models.CharField(max_length=100, blank=True)
    accuracy = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.result} ({self.accuracy:.2f}%)"