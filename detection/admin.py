from django.contrib import admin
from .models import Detection


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ['user', 'is_cracked', 'xception_confidence', 'created_at']
    list_filter = ['is_cracked', 'created_at']
    search_fields = ['user__username']
    readonly_fields = ['user', 'image', 'is_cracked', 'xception_confidence', 'created_at']
    
    def has_add_permission(self, request):
        return False
