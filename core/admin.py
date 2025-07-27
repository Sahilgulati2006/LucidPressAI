from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, UserActivity

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ('email', 'username', 'first_name', 'last_name', 'is_staff', 'is_active', 'articles_analyzed', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'date_joined', 'last_analysis_date')
    search_fields = ('email', 'username', 'first_name', 'last_name')
    ordering = ('-date_joined',)
    
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {'fields': ('articles_analyzed', 'last_analysis_date')}),
    )
    
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {'fields': ()}),
    )

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    list_display = ('user', 'activity_type', 'timestamp', 'get_details_summary')
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
