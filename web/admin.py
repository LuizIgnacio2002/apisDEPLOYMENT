from django.contrib import admin

from .models import Detection, Pigeon, Recognition

class DetectionAdmin(admin.ModelAdmin):

    list_display = ('admin_image', 'datetime', 'most_confident_label', 'confidence', 'latitude', 'longitude')

    ordering = ('-datetime',)

    readonly_fields = ('admin_image',)

admin.site.register(Detection, DetectionAdmin)
admin.site.register(Pigeon)
admin.site.register(Recognition)
