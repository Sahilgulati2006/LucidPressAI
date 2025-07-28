from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, UserActivity
from django.utils.html import format_html

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'username', 'first_name', 'last_name', 'is_staff', 'is_active', 'articles_analyzed', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'date_joined', 'last_analysis_date')
    search_fields = ('email', 'username', 'first_name', 'last_name')
    ordering = ('-date_joined',)

    # Remove 'last_login' from editable fields
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('articles_analyzed', 'last_analysis_date')}),
    )

    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ()}),
    )

    # Make last_login read-only (display only)
    readonly_fields = UserAdmin.readonly_fields + ('last_login',)

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ('user', 'activity_type', 'timestamp', 'get_details_summary', 'get_article_url')
    list_filter = ('activity_type', 'timestamp', 'user')
    search_fields = ('user__email', 'user__username', 'activity_type')
    ordering = ('-timestamp',)
    readonly_fields = ('timestamp',)

    def get_details_summary(self, obj):
        if obj.details:
            if obj.activity_type == 'article_analysis':
                return f"URL: {obj.details.get('url', 'N/A')[:50]}..."
            elif obj.activity_type == 'login':
                return f"Method: {obj.details.get('login_method', 'N/A')}"
            elif obj.activity_type == 'registration':
                return f"Method: {obj.details.get('registration_method', 'N/A')}"
            else:
                return str(obj.details)[:50]
        return 'No details'
    get_details_summary.short_description = 'Details'

    def get_article_url(self, obj):
        url = obj.details.get('url') if obj.details else None
        if url:
            return format_html('<a href="{}" target="_blank">{}</a>', url, url[:40] + ('...' if len(url) > 40 else ''))
        return '-'
    get_article_url.short_description = 'Article URL'
