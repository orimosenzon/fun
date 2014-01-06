from django.db import models

# Create your models here.
class myline (models.Model):
    t = models.CharField(max_length=200)

