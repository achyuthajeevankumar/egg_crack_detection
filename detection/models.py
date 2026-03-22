from django.db import models
from users.models import User


class Detection(models.Model):
    """Egg crack detection results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detections')
    image = models.ImageField(upload_to='detections/%Y/%m/%d/')
    is_cracked = models.BooleanField()
    

    # Xception Model Results
    xception_accuracy = models.FloatField()
    xception_confidence = models.FloatField()
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'detections'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {'Cracked' if self.is_cracked else 'Not Cracked'} - {self.created_at}"
