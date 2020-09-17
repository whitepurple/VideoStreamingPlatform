from django.contrib import admin

# Register your models here.

from .models import Stream, Face


@admin.register(Stream)
class StreamAdmin(admin.ModelAdmin):
    list_display = ("__str__", "started_at", "is_live")
    readonly_fields = ("hls_url",)

@admin.register(Face)
class FaceAdmin(admin.ModelAdmin):
    list_display = ("__str__", "streamer", "is_registerd")
    readonly_fields = ("embeding",)