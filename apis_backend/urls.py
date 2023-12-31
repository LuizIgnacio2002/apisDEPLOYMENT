"""
URL configuration for apis_backend project.

The urlpatterns list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin

from django.urls import include, path
from rest_framework import routers
from web import views
from django.conf import settings
from django.conf.urls.static import static
from apis_backend.views import home


router = routers.DefaultRouter()
router.register(r'detections', views.DetectionViewSet)
router.register(r'pigeons', views.PigeonViewSet)
router.register(r'recognitions', views.RecognitionViewSet)

admin.site.site_title = "APIS"
admin.site.site_header = "APIS"

urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    path('admin/', admin.site.urls),
    path('web/', include('web.urls')),
    path('home/', home, name='home'),
]



urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)