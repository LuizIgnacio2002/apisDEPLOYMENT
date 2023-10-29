from django.contrib.auth.models import User, Group
from rest_framework import serializers

from .models import *

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']

class DetectionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Detection
        fields = ['datetime', 'frame', 'video', 'latitude', 'longitude']


class PigeonSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Pigeon
        fields = ['name', 'description', 'image']


class RecognitionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Recognition
        fields = ['detection', 'pigeon', 'accuracy']