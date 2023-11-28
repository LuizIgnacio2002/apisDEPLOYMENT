from django.db import models
from django.core.validators import FileExtensionValidator
from PIL import Image
import numpy as np
import cv2
from django.utils.safestring import mark_safe
from django.core.files.base import ContentFile
import io
import base64
from django.core.files.uploadedfile import SimpleUploadedFile  # Agrega esta importación


class Detection(models.Model):
    datetime = models.DateTimeField("Date and time of detection", auto_now_add=True, null=True)

    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True,  null=True)
    image64 = models.TextField(null=True, blank=True)  # Campo para almacenar la imagen en formato base64
    frame = models.ImageField(upload_to="frames/", null=True, blank=True)
    frame_image_detected = models.ImageField(upload_to="frames/", null=True, blank=True)
    #video = models.FileField(upload_to="videos/", null=True,
    #validators=[FileExtensionValidator(allowed_extensions=['MOV', 'avi', 'mp4', 'webm', 'mkv'])])
    latitude = models.CharField(max_length=15, null=True)
    longitude = models.CharField(max_length=15, null=True)
    
    # New fields for detection results
    most_confident_label = models.CharField(max_length=50, blank=True, null=True)
    confidence = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Si hay una imagen en el campo image64, la procesamos antes de guardarla en frame
        if self.image64:
            self.process_image_from_base64()

        super().save(*args, **kwargs)


    def process_image_from_base64(self):
        try:
            # Decodificar la imagen en base64 a una cadena binaria
            binary_image = base64.b64decode(self.image64)

            # Crear un objeto SimpleUploadedFile para guardar la imagen en el campo 'frame'
            image_name = f"original_{self.pk}.jpg"  # Cambiar la extensión según el formato
            image_file = SimpleUploadedFile(image_name, binary_image, content_type="image/jpeg")

            # Guardar la imagen en el campo 'frame'
            self.frame.save(image_name, image_file, save=False)
            
            # Procesar la imagen como se hacía antes
            self.process_image_for_detection()

        except Exception as e:
            # Manejar cualquier error que pueda ocurrir durante el proceso
            print(f"Error al procesar la imagen desde image64: {e}")


    def admin_image(self):
        return mark_safe('<img src="{url}" width="{width}" height={height} />'.format(
            url=self.frame.url,
            width=100,
            height=100,
            ))
    admin_image.short_description = 'Image'
    admin_image.allow_tags = True

    def admin_image_detected(self):
        return mark_safe('<img src="{url}" width="{width}" height={height} />'.format(
            url=self.frame_image_detected.url,
            width=100,
            height=100,
            ))
    admin_image_detected.short_description = 'Image Detected'
    admin_image_detected.allow_tags = True

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

                # Draw bounding box on the original image
                box = detection[3:7] * [width, height, width, height]
                x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(open_cv_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(open_cv_image, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
                cv2.putText(open_cv_image, self.most_confident_label, (x_start, y_start - 25), 1, 1.2, (255, 0, 0), 2)

                # Save the modified image to frame_image_detected
                buffer = io.BytesIO()
                image_with_box = Image.fromarray(open_cv_image)
                image_with_box.save(buffer, format='JPEG')

                # Convert the NumPy array to RGB format
                image_with_box_rgb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

                # Save the modified image to frame_image_detected
                buffer = io.BytesIO()
                image_with_box_pil = Image.fromarray(image_with_box_rgb)
                image_with_box_pil.save(buffer, format='JPEG')
                self.frame_image_detected.save('detected_{}.jpg'.format(self.pk), ContentFile(buffer.getvalue()), save=False)


        # Save the updated Detection instance
        super().save(update_fields=['most_confident_label', 'confidence', 'frame_image_detected'])


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
    
