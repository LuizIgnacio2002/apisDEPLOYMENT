from django.contrib import admin

from .models import Detection, Pigeon, Recognition

class DetectionAdmin(admin.ModelAdmin):

    list_display = ('admin_image', 'admin_image_detected', 'datetime', 'most_confident_label', 'confidence', 'latitude', 'longitude')

    ordering = ('-datetime',)

    readonly_fields = ('admin_image', 'admin_image_detected')

admin.site.register(Detection, DetectionAdmin)
admin.site.register(Pigeon)
admin.site.register(Recognition)
