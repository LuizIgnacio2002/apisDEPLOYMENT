from django.db import models
from django.core.validators import FileExtensionValidator
from PIL import Image
import numpy as np
import cv2

class Detection(models.Model):
    datetime = models.DateTimeField("Date and time of detection", auto_now_add=True, null=True)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True,  null=True)
    frame = models.ImageField(upload_to="frames/", null=True)
    #video = models.FileField(upload_to="videos/", null=True,
    #validators=[FileExtensionValidator(allowed_extensions=['MOV', 'avi', 'mp4', 'webm', 'mkv'])])
    latitude = models.CharField(max_length=15, null=True)
    longitude = models.CharField(max_length=15, null=True)
    
    # New fields for detection results
    most_confident_label = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return str(self.datetime)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.frame:
            self.process_image_for_detection()

    def process_image_for_detection(self):
        # Model architecture and weights
        prototxt = "web/MobileNetSSD_deploy.prototxt.txt"
        model = "web/MobileNetSSD_deploy.caffemodel"

        # Load the model
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        open_cv_image = cv2.imread(self.frame.path)

        # Preprocess the image for object detection
        height, width, _ = open_cv_image.shape
        image_resized = cv2.resize(open_cv_image, (300, 300))
        blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

        # Perform detections
        net.setInput(blob)
        detections = net.forward()

        # Class labels
        classes = {0:"background", 1:"aeroplane", 2:"bicycle",
        3:"bird", 4:"boat",
        5:"bottle", 6:"bus",
        7:"car", 8:"cat",
        9:"chair", 10:"cow",
        11:"diningtable", 12:"dog",
        13:"horse", 14:"motorbike",
        15:"person", 16:"pottedplant",
        17:"sheep", 18:"sofa",
        19:"train", 20:"tvmonitor"}

        max_confidence = 0
        for detection in detections[0][0]:
            if detection[2] > max_confidence:
                max_confidence = detection[2]
                self.most_confident_label = classes[int(detection[1])]
                self.confidence = float(detection[2])

        super().save(update_fields=['most_confident_label', 'confidence'])


class Pigeon(models.Model):
    name = models.CharField(max_length = 15)
    description = models.CharField(max_length = 250)
    image = models.ImageField(upload_to = "image/", null = True) 

    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name = "Ave"
        verbose_name_plural = "Aves"

class Recognition(models.Model):
    detection = models.ForeignKey(Detection, on_delete = models.CASCADE)
    pigeon = models.ForeignKey(Pigeon, on_delete = models.CASCADE)
    accuracy = models.FloatField(null = True, blank = True)

    def __str__(self):
        return self.detection
    
    class Meta:
        verbose_name = "Reconocimiento"
        verbose_name_plural = "Reconocimientos"
    
