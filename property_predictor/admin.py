from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Avg, Count, Sum
from django.urls import path
from django.shortcuts import render
from django.http import HttpResponse
from django.utils import timezone
from datetime import timedelta
import csv

from .models import (
    Location, PropertyType, PropertyPrediction, 
    ModelMetrics, MarketInsight
)

admin.site.site_header = "Nairobi Property Predictor Admin"
admin.site.site_title = "Property Predictor Admin"
admin.site.index_title = "Property Price Prediction Management"


@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    list_display = ['name', 'tier_badge', 'total_predictions', 'avg_price', 'description_short']
    list_filter = ['tier']
    search_fields = ['name', 'description']
    ordering = ['name']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'tier', 'description')
        }),
        ('Geographical Data', {
            'fields': ('latitude', 'longitude'),
            'classes': ('collapse',)
        }),
    )
    
    def tier_badge(self, obj):
        colors = {
            'Tier1_Premium': '#f59e0b',
            'Tier2_Upscale': '#3b82f6',
            'Tier3_Standard': '#8b5cf6'
        }
        color = colors.get(obj.tier, '#6b7280')
        tier_name = obj.tier.replace('Tier', '').replace('_', ' ')
        return format_html(
            '<span style="background: {}; color: white; padding: 4px 12px; '
            'border-radius: 12px; font-weight: 600; font-size: 0.85rem;">{}</span>',
            color, tier_name
        )
    tier_badge.short_description = 'Tier'
    
    def total_predictions(self, obj):
        count = PropertyPrediction.objects.filter(location=obj).count()
        return format_html(
            '<span style="font-weight: 700; color: #2563eb;">{}</span>',
            count
        )
    total_predictions.short_description = 'Predictions'
    
    def avg_price(self, obj):
        avg = PropertyPrediction.objects.filter(
            location=obj
        ).aggregate(Avg('predicted_price'))['predicted_price__avg']
        
        if avg:
            return format_html(
                '<span style="font-weight: 700; color: #10b981;">KSh {:,.0f}</span>',
                avg
            )
        return '-'
    avg_price.short_description = 'Avg Price'
    
    def description_short(self, obj):
        if obj.description:
            return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
        return '-'
    description_short.short_description = 'Description'


@admin.register(PropertyType)
class PropertyTypeAdmin(admin.ModelAdmin):
    list_display = ['name', 'total_predictions', 'avg_price', 'description_short']
    search_fields = ['name', 'description']
    ordering = ['name']
    
    def total_predictions(self, obj):
        count = PropertyPrediction.objects.filter(property_type=obj).count()
        return format_html(
            '<span style="font-weight: 700; color: #2563eb;">{}</span>',
            count
        )
    total_predictions.short_description = 'Predictions'
    
    def avg_price(self, obj):
        avg = PropertyPrediction.objects.filter(
            property_type=obj
        ).aggregate(Avg('predicted_price'))['predicted_price__avg']
        
        if avg:
            return format_html(
                '<span style="font-weight: 700; color: #10b981;">KSh {:,.0f}</span>',
                avg
            )
        return '-'
    avg_price.short_description = 'Avg Price'
    
    def description_short(self, obj):
        if obj.description:
            return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description
        return '-'
    description_short.short_description = 'Description'


@admin.register(PropertyPrediction)
class PropertyPredictionAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'property_type', 'location', 'bedrooms', 'bathrooms',
        'price_display', 'confidence_badge', 'rating_stars', 'created_display'
    ]
    list_filter = [
        'property_type', 'location', 'created_at', 
        'user_rating', ('confidence_score', admin.EmptyFieldListFilter)
    ]
    search_fields = ['location__name', 'property_type__name', 'user_ip']
    readonly_fields = [
        'predicted_price', 'price_lower_bound', 'price_upper_bound',
        'confidence_score', 'created_at', 'updated_at', 'user_ip', 'session_id'
    ]
    date_hierarchy = 'created_at'
    ordering = ['-created_at']
    
    fieldsets = (
        ('Property Details', {
            'fields': ('property_type', 'location', 'bedrooms', 'bathrooms', 
                      'house_size', 'land_size')
        }),
        ('Prediction Results', {
            'fields': ('predicted_price', 'price_lower_bound', 'price_upper_bound', 
                      'confidence_score'),
            'classes': ('wide',)
        }),
        ('User Feedback', {
            'fields': ('user_rating', 'user_comments'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at', 'user_ip', 'session_id'),
            'classes': ('collapse',)
        }),
    )
    
    actions = ['export_to_csv', 'calculate_statistics']
    
    def price_display(self, obj):
        if obj.predicted_price:
            return format_html(
                '<div style="font-weight: 800; color: #2563eb; font-size: 1.1rem;">'
                'KSh {:,.0f}</div>'
                '<small style="color: #6b7280;">Range: {:,.0f} - {:,.0f}</small>',
                obj.predicted_price,
                obj.price_lower_bound or 0,
                obj.price_upper_bound or 0
            )
        return '-'
    price_display.short_description = 'Predicted Price'
    
    def confidence_badge(self, obj):
        if obj.confidence_score:
            score = obj.confidence_score * 100
            if score >= 85:
                color = '#10b981'  
            elif score >= 70:
                color = '#f59e0b'  
            else:
                color = '#ef4444'  
            
            return format_html(
                '<span style="background: {}; color: white; padding: 4px 10px; '
                'border-radius: 12px; font-weight: 600;">{:.0f}%</span>',
                color, score
            )
        return '-'
    confidence_badge.short_description = 'Confidence'
    
    def rating_stars(self, obj):
        if obj.user_rating:
            stars = '⭐' * obj.user_rating
            return format_html('<span style="font-size: 1.2rem;">{}</span>', stars)
        return format_html('<span style="color: #d1d5db;">No rating</span>')
    rating_stars.short_description = 'User Rating'
    
    def created_display(self, obj):
        return format_html(
            '<div style="font-size: 0.9rem;">{}</div>'
            '<small style="color: #6b7280;">{} ago</small>',
            obj.created_at.strftime('%Y-%m-%d %H:%M'),
            self.time_since(obj.created_at)
        )
    created_display.short_description = 'Created'
    
    def time_since(self, dt):
        diff = timezone.now() - dt
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''}"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''}"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"{minutes} min{'s' if minutes > 1 else ''}"
        else:
            return "Just now"
    
    def export_to_csv(self, request, queryset):
        """Export selected predictions to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="predictions.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'ID', 'Property Type', 'Location', 'Bedrooms', 'Bathrooms',
            'House Size', 'Land Size', 'Predicted Price', 'Lower Bound',
            'Upper Bound', 'Confidence', 'Rating', 'Created At'
        ])
        
        for pred in queryset:
            writer.writerow([
                pred.id,
                pred.property_type.name if pred.property_type else '',
                pred.location.name if pred.location else '',
                pred.bedrooms,
                pred.bathrooms,
                pred.house_size,
                pred.land_size,
                pred.predicted_price,
                pred.price_lower_bound,
                pred.price_upper_bound,
                pred.confidence_score,
                pred.user_rating or '',
                pred.created_at.strftime('%Y-%m-%d %H:%M:%S')
            ])
        
        return response
    export_to_csv.short_description = "Export selected to CSV"
    
    def calculate_statistics(self, request, queryset):
        """Calculate statistics for selected predictions"""
        stats = queryset.aggregate(
            total=Count('id'),
            avg_price=Avg('predicted_price'),
            avg_confidence=Avg('confidence_score'),
            avg_rating=Avg('user_rating')
        )
        
        message = (
            f"Statistics for {stats['total']} predictions: "
            f"Avg Price: KSh {stats['avg_price']:,.0f}, "
            f"Avg Confidence: {stats['avg_confidence']*100:.1f}%, "
            f"Avg Rating: {stats['avg_rating']:.1f}/5"
        )
        
        self.message_user(request, message)
    calculate_statistics.short_description = "Calculate statistics"
    
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('analytics/', self.admin_site.admin_view(self.analytics_view), 
                 name='prediction_analytics'),
        ]
        return custom_urls + urls
    
    def analytics_view(self, request):
        """Custom analytics view"""
        # Last 30 days
        thirty_days_ago = timezone.now() - timedelta(days=30)
        recent_predictions = PropertyPrediction.objects.filter(
            created_at__gte=thirty_days_ago
        )
        
    
        stats = {
            'total_predictions': PropertyPrediction.objects.count(),
            'recent_predictions': recent_predictions.count(),
            'avg_price': PropertyPrediction.objects.aggregate(
                Avg('predicted_price')
            )['predicted_price__avg'] or 0,
            'avg_confidence': PropertyPrediction.objects.aggregate(
                Avg('confidence_score')
            )['confidence_score__avg'] or 0,
        }
        
        
        top_locations = PropertyPrediction.objects.values(
            'location__name'
        ).annotate(
            count=Count('id'),
            avg_price=Avg('predicted_price')
        ).order_by('-count')[:5]
        
        
        top_types = PropertyPrediction.objects.values(
            'property_type__name'
        ).annotate(
            count=Count('id'),
            avg_price=Avg('predicted_price')
        ).order_by('-count')[:5]
        
        context = {
            'title': 'Prediction Analytics',
            'stats': stats,
            'top_locations': top_locations,
            'top_types': top_types,
            'opts': self.model._meta,
        }
        
        return render(request, 'admin/prediction_analytics.html', context)


@admin.register(ModelMetrics)
class ModelMetricsAdmin(admin.ModelAdmin):
    list_display = [
        'model_name', 'version', 'performance_badge', 
        'total_samples', 'feature_count', 'is_active', 'training_date'
    ]
    list_filter = ['is_active', 'training_date', 'model_name']
    search_fields = ['model_name', 'version', 'notes']
    readonly_fields = [
        'train_mae', 'test_mae', 'train_rmse', 'test_rmse',
        'train_r2', 'test_r2', 'training_date'
    ]
    ordering = ['-training_date']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('model_name', 'version', 'is_active', 'training_date')
        }),
        ('Training Metrics', {
            'fields': ('train_mae', 'train_rmse', 'train_r2'),
            'classes': ('wide',)
        }),
        ('Test Metrics', {
            'fields': ('test_mae', 'test_rmse', 'test_r2'),
            'classes': ('wide',)
        }),
        ('Additional Information', {
            'fields': ('total_samples', 'feature_count', 'notes'),
            'classes': ('collapse',)
        }),
    )
    
    def performance_badge(self, obj):
        r2_score = obj.test_r2
        if r2_score >= 0.9:
            color = '#10b981'
            label = 'Excellent'
        elif r2_score >= 0.8:
            color = '#3b82f6'
            label = 'Good'
        elif r2_score >= 0.7:
            color = '#f59e0b'
            label = 'Fair'
        else:
            color = '#ef4444'
            label = 'Poor'
        
        return format_html(
            '<div style="display: inline-block;">'
            '<span style="background: {}; color: white; padding: 4px 12px; '
            'border-radius: 12px; font-weight: 600; margin-right: 8px;">{}</span>'
            '<span style="font-weight: 700; color: #2563eb;">R²: {:.3f}</span>'
            '</div>',
            color, label, r2_score
        )
    performance_badge.short_description = 'Performance'


@admin.register(MarketInsight)
class MarketInsightAdmin(admin.ModelAdmin):
    list_display = [
        'location', 'property_type', 'avg_price_display',
        'price_range', 'trend_badge', 'sample_count', 'last_updated'
    ]
    list_filter = ['price_trend', 'location', 'property_type', 'last_updated']
    search_fields = ['location__name', 'property_type__name']
    readonly_fields = ['last_updated']
    ordering = ['-last_updated']
    
    fieldsets = (
        ('Market Segment', {
            'fields': ('location', 'property_type')
        }),
        ('Price Statistics', {
            'fields': ('avg_price', 'median_price', 'min_price', 'max_price', 'std_dev'),
            'classes': ('wide',)
        }),
        ('Trend Information', {
            'fields': ('price_trend', 'sample_count', 'last_updated')
        }),
    )
    
    def avg_price_display(self, obj):
        return format_html(
            '<span style="font-weight: 800; color: #2563eb; font-size: 1.1rem;">'
            'KSh {:,.0f}</span>',
            obj.avg_price
        )
    avg_price_display.short_description = 'Avg Price'
    
    def price_range(self, obj):
        return format_html(
            '<small style="color: #6b7280;">KSh {:,.0f} - {:,.0f}</small>',
            obj.min_price, obj.max_price
        )
    price_range.short_description = 'Range'
    
    def trend_badge(self, obj):
        colors = {
            'increasing': '#10b981',
            'stable': '#3b82f6',
            'decreasing': '#ef4444'
        }
        icons = {
            'increasing': '↑',
            'stable': '→',
            'decreasing': '↓'
        }
        
        color = colors.get(obj.price_trend, '#6b7280')
        icon = icons.get(obj.price_trend, '-')
        
        return format_html(
            '<span style="background: {}; color: white; padding: 4px 12px; '
            'border-radius: 12px; font-weight: 600;">{} {}</span>',
            color, icon, obj.price_trend.title()
        )
    trend_badge.short_description = 'Trend'


# Custom Admin Dashboard
class CustomAdminSite(admin.AdminSite):
    def index(self, request, extra_context=None):
        extra_context = extra_context or {}
        
        # Dashboard statistics
        thirty_days_ago = timezone.now() - timedelta(days=30)
        
        extra_context['dashboard_stats'] = {
            'total_predictions': PropertyPrediction.objects.count(),
            'predictions_30d': PropertyPrediction.objects.filter(
                created_at__gte=thirty_days_ago
            ).count(),
            'total_locations': Location.objects.count(),
            'total_property_types': PropertyType.objects.count(),
            'avg_price': PropertyPrediction.objects.aggregate(
                Avg('predicted_price')
            )['predicted_price__avg'] or 0,
            'avg_confidence': PropertyPrediction.objects.aggregate(
                Avg('confidence_score')
            )['confidence_score__avg'] or 0,
        }
        
        return super().index(request, extra_context)


# Inline admin for related models
class PropertyPredictionInline(admin.TabularInline):
    model = PropertyPrediction
    extra = 0
    fields = ['property_type', 'location', 'predicted_price', 'created_at']
    readonly_fields = ['predicted_price', 'created_at']
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False
