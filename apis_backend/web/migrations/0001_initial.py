# Generated by Django 4.2.6 on 2023-10-29 01:26

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Detection',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('latitude', models.CharField(max_length=15)),
                ('longitude', models.CharField(max_length=15)),
                ('datetime', models.DateTimeField(verbose_name='Date and time of detection')),
                ('frame', models.ImageField(null=True, upload_to='frames/')),
                ('video', models.FileField(null=True, upload_to='videos/', validators=[django.core.validators.FileExtensionValidator(allowed_extensions=['MOV', 'avi', 'mp4', 'webm', 'mkv'])])),
            ],
        ),
        migrations.CreateModel(
            name='Pigeon',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=15)),
                ('description', models.CharField(max_length=250)),
                ('image', models.ImageField(null=True, upload_to='image/')),
            ],
        ),
        migrations.CreateModel(
            name='Recognition',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy', models.FloatField(blank=True, null=True)),
                ('detection', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='web.detection')),
                ('pigeon', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='web.pigeon')),
            ],
        ),
    ]
