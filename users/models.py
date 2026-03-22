from django.db import models
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    """Custom User model"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('admin', 'Admin'),
    ]
    
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    full_name = models.CharField(max_length=255, blank=True)
    mobile = models.CharField(max_length=10, blank=True)
    status = models.CharField(max_length=10, default='waiting')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'users'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.username


class LoginActivity(models.Model):
    """Track user login activities"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='login_activities')
    ip_address = models.GenericIPAddressField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'login_activities'
        ordering = ['-timestamp']
        verbose_name_plural = 'Login Activities'
    
    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"
