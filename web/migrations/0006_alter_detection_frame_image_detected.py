# Generated by Django 4.2.6 on 2023-11-27 21:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0005_detection_frame_image_detected'),
    ]

    operations = [
        migrations.AlterField(
            model_name='detection',
            name='frame_image_detected',
            field=models.ImageField(blank=True, null=True, upload_to='frames/'),
        ),
    ]