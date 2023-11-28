from django.urls import path, include
from django.shortcuts import render

def home(request):
    return render(request, 'core/home.html') 