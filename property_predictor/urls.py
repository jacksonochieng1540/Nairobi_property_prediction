from django.urls import path
from . import views

urlpatterns = [
    # Web interface
    path('', views.home, name='home'),
    path('result/<int:pk>/', views.prediction_result, name='prediction_result'),
    path('result/<int:pk>/feedback/', views.submit_feedback, name='submit_feedback'),
    path('history/', views.history, name='history'),
    path('insights/', views.market_insights, name='market_insights'),
    path('compare/', views.compare_properties, name='compare_properties'),
    
    # API endpoints
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/locations/', views.api_locations, name='api_locations'),
    path('api/property-types/', views.api_property_types, name='api_property_types'),
    path('api/market-stats/', views.api_market_stats, name='api_market_stats'),
]