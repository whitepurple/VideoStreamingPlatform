# Generated by Django 3.0.3 on 2020-09-20 16:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('testapp', '0004_auto_20200920_1510'),
    ]

    operations = [
        migrations.AlterField(
            model_name='face',
            name='profile',
            field=models.ImageField(null=True, upload_to='profile/'),
        ),
    ]
