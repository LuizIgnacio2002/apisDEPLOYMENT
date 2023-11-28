from django.contrib.auth.models import User, Group
from rest_framework import serializers

from .models import *

class DetectionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Detection
        fields = ['datetime', 'frame', 'latitude', 'longitude', 'most_confident_label', 'confidence', 'image64']

class PigeonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta: 
        model = Pigeon
        fields = ['name', 'description', 'image']

class RecognitionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Recognition
        fields = ['detection', 'pigeon', 'accuracy']
