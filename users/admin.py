from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, LoginActivity


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['username', 'email']
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Custom Fields', {'fields': ('role',)}),
    )


@admin.register(LoginActivity)
class LoginActivityAdmin(admin.ModelAdmin):
    list_display = ['user', 'ip_address', 'timestamp']
    list_filter = ['timestamp']
    search_fields = ['user__username', 'ip_address']
    readonly_fields = ['user', 'ip_address', 'timestamp']
