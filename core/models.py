from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone

class User(AbstractUser):
    """Custom user model with additional fields"""
    email = models.EmailField(unique=True)
    date_joined = models.DateTimeField(default=timezone.now)
    last_login = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Track user activity
    articles_analyzed = models.IntegerField(default=0)
    last_analysis_date = models.DateTimeField(blank=True, null=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    def __str__(self):
        return self.email
    
    def increment_articles_analyzed(self):
        """Increment the count of articles analyzed by this user"""
        self.articles_analyzed += 1
        self.last_analysis_date = timezone.now()
        self.save()

class UserActivity(models.Model):
    """Model to track user activities"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    activity_type = models.CharField(max_length=50)  # 'login', 'article_analysis', 'registration'
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.email} - {self.activity_type} - {self.timestamp}"
