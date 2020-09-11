from django.contrib import admin
from . import models

@admin.register(models.DiceUser)
class UserAdmin(admin.ModelAdmin):

    list_display = (
        'username',
        'date_joined',
    )

    list_display_links = (
        'username',
    )