from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from web.serializers import *
from .models import *
from rest_framework.permissions import AllowAny
import random as rd
import base64
import json
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse

class DetectionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Detections to be viewed or edited.
    """
    queryset = Detection.objects.all().order_by('-datetime')
    serializer_class = DetectionSerializer
    permission_classes = [AllowAny]

class PigeonViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Pigeons to be viewed or edited.
    """
    queryset = Pigeon.objects.all()
    serializer_class = PigeonSerializer
    permission_classes = [AllowAny]

class RecognitionViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows Recognitions to be viewed or edited.
    """
    queryset = Recognition.objects.all()
    serializer_class = RecognitionSerializer
    permission_classes = [AllowAny]

@csrf_exempt
@require_POST
def send_img(request):
    if request.method == 'POST':
        # data = request.POST #json
        data = json.loads(request.body)
        imagen_base64 = data.get('frame', None)

        if imagen_base64 is not None:
            try:
                imagen_decodificada = base64.b64decode(imagen_base64)
                imagen_temporal = ContentFile(imagen_decodificada)

                nuevo_registro = Detection(longitude='15', latitude='23')
                nuevo_registro.frame.save(f'{rd.randint(10, 10000)}.jpg', imagen_temporal)

                return JsonResponse({
                    'most_confident_label' : nuevo_registro.most_confident_label,
                    'confidence' : nuevo_registro.confidence
                })
            except Exception as e:
                return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Método no permitido'}, status=405)