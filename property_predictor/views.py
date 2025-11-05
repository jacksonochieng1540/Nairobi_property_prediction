# nairobi_property_predictor/property_predictor/views.py

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.db.models import Avg, Count, Min, Max
from django.core.paginator import Paginator
import json

from .models import PropertyPrediction, Location, PropertyType, MarketInsight
from .forms import PropertyPredictionForm, FeedbackForm
from .ml_models.prediction_service import get_prediction_service


def home(request):
    """Home page with prediction form"""
    if request.method == 'POST':
        form = PropertyPredictionForm(request.POST)
        if form.is_valid():
            # Save the input data
            prediction = form.save(commit=False)
            
            # Get user IP
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip = x_forwarded_for.split(',')[0]
            else:
                ip = request.META.get('REMOTE_ADDR')
            prediction.user_ip = ip
            prediction.session_id = request.session.session_key
            
            # Prepare data for prediction
            property_data = {
                'property_type': prediction.property_type.name,
                'location': prediction.location.name,
                'bedrooms': prediction.bedrooms,
                'bathrooms': prediction.bathrooms,
                'house_size': prediction.house_size,
                'land_size': prediction.land_size
            }
            
            # Make prediction
            service = get_prediction_service()
            result = service.predict(property_data)
            
            if result['success']:
                # Save prediction results
                prediction.predicted_price = result['predicted_price']
                prediction.price_lower_bound = result['lower_bound']
                prediction.price_upper_bound = result['upper_bound']
                prediction.confidence_score = result['confidence_score']
                prediction.save()
                
                # Redirect to results page
                return redirect('prediction_result', pk=prediction.pk)
            else:
                messages.error(request, f"Prediction failed: {result['error']}")
    else:
        form = PropertyPredictionForm()
    
    # Get statistics for display
    recent_predictions = PropertyPrediction.objects.select_related(
        'property_type', 'location'
    ).order_by('-created_at')[:5]
    
    stats = {
        'total_predictions': PropertyPrediction.objects.count(),
        'avg_price': PropertyPrediction.objects.aggregate(
            Avg('predicted_price')
        )['predicted_price__avg'] or 0
    }
    
    context = {
        'form': form,
        'recent_predictions': recent_predictions,
        'stats': stats
    }
    return render(request, 'property_predictor/home.html', context)


def prediction_result(request, pk):
    """Display prediction results"""
    prediction = get_object_or_404(
        PropertyPrediction.objects.select_related('property_type', 'location'),
        pk=pk
    )
    
    # Get similar properties
    similar_props = PropertyPrediction.objects.filter(
        location=prediction.location,
        property_type=prediction.property_type
    ).exclude(pk=pk).order_by('-created_at')[:5]
    
    # Calculate price per sqm if applicable
    price_per_sqm = None
    if prediction.house_size > 0:
        price_per_sqm = float(prediction.predicted_price) / prediction.house_size
    
    # Get market insights
    market_insight = MarketInsight.objects.filter(
        location=prediction.location,
        property_type=prediction.property_type
    ).first()
    
    context = {
        'prediction': prediction,
        'similar_properties': similar_props,
        'price_per_sqm': price_per_sqm,
        'market_insight': market_insight,
        'feedback_form': FeedbackForm()
    }
    return render(request, 'property_predictor/result.html', context)


def submit_feedback(request, pk):
    """Submit user feedback on prediction"""
    if request.method == 'POST':
        prediction = get_object_or_404(PropertyPrediction, pk=pk)
        form = FeedbackForm(request.POST, instance=prediction)
        
        if form.is_valid():
            form.save()
            messages.success(request, 'Thank you for your feedback!')
        else:
            messages.error(request, 'Error submitting feedback')
    
    return redirect('prediction_result', pk=pk)


def history(request):
    """View prediction history"""
    predictions = PropertyPrediction.objects.select_related(
        'property_type', 'location'
    ).order_by('-created_at')
    
    # Pagination
    paginator = Paginator(predictions, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj
    }
    return render(request, 'property_predictor/history.html', context)


def market_insights(request):
    """Display market insights and analytics"""
    # Get location-wise statistics
    location_stats = PropertyPrediction.objects.values(
        'location__name', 'location__tier'
    ).annotate(
        avg_price=Avg('predicted_price'),
        min_price=Min('predicted_price'),
        max_price=Max('predicted_price'),
        count=Count('id')
    ).order_by('-avg_price')
    
    # Get property type statistics
    type_stats = PropertyPrediction.objects.values(
        'property_type__name'
    ).annotate(
        avg_price=Avg('predicted_price'),
        count=Count('id')
    ).order_by('-avg_price')
    
    # Get all market insights
    insights = MarketInsight.objects.select_related(
        'location', 'property_type'
    ).order_by('-last_updated')
    
    context = {
        'location_stats': location_stats,
        'type_stats': type_stats,
        'insights': insights
    }
    return render(request, 'property_predictor/market_insights.html', context)


def compare_properties(request):
    """Compare multiple properties"""
    if request.method == 'POST':
        # Get form data
        properties = []
        for i in range(1, 3):  # Compare 2 properties
            prop_data = {
                'property_type': request.POST.get(f'property{i}_type'),
                'location': request.POST.get(f'property{i}_location'),
                'bedrooms': int(request.POST.get(f'property{i}_bedrooms', 0)),
                'bathrooms': int(request.POST.get(f'property{i}_bathrooms', 0)),
                'house_size': float(request.POST.get(f'property{i}_house_size', 0)),
                'land_size': float(request.POST.get(f'property{i}_land_size', 0))
            }
            properties.append(prop_data)
        
        # Make predictions
        service = get_prediction_service()
        results = service.predict_batch(properties)
        
        context = {
            'properties': properties,
            'results': results
        }
        return render(request, 'property_predictor/compare_results.html', context)
    
    # GET request - show form
    locations = Location.objects.all()
    property_types = PropertyType.objects.all()
    
    context = {
        'locations': locations,
        'property_types': property_types
    }
    return render(request, 'property_predictor/compare.html', context)


# API Views
@require_http_methods(["POST"])
def api_predict(request):
    """API endpoint for predictions"""
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['property_type', 'location', 'bedrooms', 'bathrooms']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }, status=400)
        
        # Make prediction
        service = get_prediction_service()
        result = service.predict(data)
        
        return JsonResponse(result)
    
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def api_locations(request):
    """API endpoint to get all locations"""
    locations = Location.objects.values('id', 'name', 'tier')
    return JsonResponse(list(locations), safe=False)


@require_http_methods(["GET"])
def api_property_types(request):
    """API endpoint to get all property types"""
    property_types = PropertyType.objects.values('id', 'name')
    return JsonResponse(list(property_types), safe=False)


@require_http_methods(["GET"])
def api_market_stats(request):
    """API endpoint for market statistics"""
    location = request.GET.get('location')
    property_type = request.GET.get('property_type')
    
    filters = {}
    if location:
        filters['location__name'] = location
    if property_type:
        filters['property_type__name'] = property_type
    
    stats = PropertyPrediction.objects.filter(**filters).aggregate(
        avg_price=Avg('predicted_price'),
        min_price=Min('predicted_price'),
        max_price=Max('predicted_price'),
        count=Count('id')
    )
    
    return JsonResponse(stats)