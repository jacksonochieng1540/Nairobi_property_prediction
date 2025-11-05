# nairobi_property_predictor/property_predictor/models.py

from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class Location(models.Model):
    """Location master data"""
    name = models.CharField(max_length=100, unique=True)
    tier = models.CharField(
        max_length=20,
        choices=[
            ('Tier1_Premium', 'Premium'),
            ('Tier2_Upscale', 'Upscale'),
            ('Tier3_Standard', 'Standard')
        ],
        default='Tier3_Standard'
    )
    latitude = models.DecimalField(
        max_digits=9, 
        decimal_places=6, 
        null=True, 
        blank=True
    )
    longitude = models.DecimalField(
        max_digits=9, 
        decimal_places=6, 
        null=True, 
        blank=True
    )
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.tier})"


class PropertyType(models.Model):
    """Property type master data"""
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    
    class Meta:
        ordering = ['name']
    
    def __str__(self):
        return self.name


class PropertyPrediction(models.Model):
    """Store prediction requests and results"""
    
    # Input features
    property_type = models.ForeignKey(
        PropertyType,
        on_delete=models.SET_NULL,
        null=True
    )
    location = models.ForeignKey(
        Location,
        on_delete=models.SET_NULL,
        null=True
    )
    bedrooms = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(20)],
        default=0
    )
    bathrooms = models.IntegerField(
        validators=[MinValueValidator(0), MaxValueValidator(20)],
        default=0
    )
    house_size = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="House size in square meters",
        default=0
    )
    land_size = models.FloatField(
        validators=[MinValueValidator(0)],
        help_text="Land size in square meters",
        default=0
    )
    
    # Prediction results
    predicted_price = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True
    )
    price_lower_bound = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True
    )
    price_upper_bound = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        null=True,
        blank=True
    )
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0), MaxValueValidator(1)],
        null=True,
        blank=True
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    
    # User feedback
    user_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        null=True,
        blank=True,
        help_text="User rating of prediction accuracy (1-5)"
    )
    user_comments = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['location', 'property_type']),
        ]
    
    def __str__(self):
        return f"{self.property_type} in {self.location} - KSh {self.predicted_price:,.0f}"
    
    @property
    def price_range_text(self):
        """Human-readable price range"""
        if self.price_lower_bound and self.price_upper_bound:
            return f"KSh {self.price_lower_bound:,.0f} - KSh {self.price_upper_bound:,.0f}"
        return "N/A"


class ModelMetrics(models.Model):
    """Store model performance metrics"""
    model_name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    
    # Metrics
    train_mae = models.FloatField()
    test_mae = models.FloatField()
    train_rmse = models.FloatField()
    test_rmse = models.FloatField()
    train_r2 = models.FloatField()
    test_r2 = models.FloatField()
    
    # Additional info
    training_date = models.DateTimeField()
    total_samples = models.IntegerField()
    feature_count = models.IntegerField()
    notes = models.TextField(blank=True)
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-training_date']
        verbose_name_plural = "Model Metrics"
    
    def __str__(self):
        return f"{self.model_name} v{self.version} - R²: {self.test_r2:.4f}"


class MarketInsight(models.Model):
    """Store market insights and statistics"""
    location = models.ForeignKey(
        Location,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    property_type = models.ForeignKey(
        PropertyType,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    
    # Statistics
    avg_price = models.DecimalField(max_digits=15, decimal_places=2)
    median_price = models.DecimalField(max_digits=15, decimal_places=2)
    min_price = models.DecimalField(max_digits=15, decimal_places=2)
    max_price = models.DecimalField(max_digits=15, decimal_places=2)
    std_dev = models.DecimalField(max_digits=15, decimal_places=2)
    
    sample_count = models.IntegerField()
    
    # Trends
    price_trend = models.CharField(
        max_length=20,
        choices=[
            ('increasing', 'Increasing'),
            ('stable', 'Stable'),
            ('decreasing', 'Decreasing')
        ],
        default='stable'
    )
    
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_updated']
        unique_together = ['location', 'property_type']
    
    def __str__(self):
        loc = self.location.name if self.location else "All Locations"
        prop = self.property_type.name if self.property_type else "All Types"
        return f"{prop} in {loc} - Avg: KSh {self.avg_price:,.0f}"
