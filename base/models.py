from attr import field
from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class csv(models.Model):
    title = models.CharField(max_length=50)
def __str__(self):
    return self.title

class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields=['username','email','password1','password2',]