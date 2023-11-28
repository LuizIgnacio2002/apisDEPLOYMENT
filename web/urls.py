from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name = "index"),
    path('sendImg/', send_img, name="sendImg"),
]