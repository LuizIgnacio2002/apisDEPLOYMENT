# Generated by Django 4.2.6 on 2023-11-28 00:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0006_alter_detection_frame_image_detected'),
    ]

    operations = [
        migrations.AddField(
            model_name='detection',
            name='image64',
            field=models.TextField(blank=True, null=True),
        ),
    ]
