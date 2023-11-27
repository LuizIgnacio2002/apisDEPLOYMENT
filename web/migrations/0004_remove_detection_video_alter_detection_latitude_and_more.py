# Generated by Django 4.2.6 on 2023-11-27 16:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('web', '0003_alter_detection_options_detection_confidence_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='detection',
            name='video',
        ),
        migrations.AlterField(
            model_name='detection',
            name='latitude',
            field=models.CharField(max_length=15, null=True),
        ),
        migrations.AlterField(
            model_name='detection',
            name='longitude',
            field=models.CharField(max_length=15, null=True),
        ),
    ]