from django import forms
from .models import PropertyPrediction, Location, PropertyType


class PropertyPredictionForm(forms.ModelForm):
    """Form for property price prediction"""
    
    class Meta:
        model = PropertyPrediction
        fields = [
            'property_type', 'location', 'bedrooms', 'bathrooms',
            'house_size', 'land_size'
        ]
        
        widgets = {
            'property_type': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'location': forms.Select(attrs={
                'class': 'form-select',
                'required': True
            }),
            'bedrooms': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0,
                'max': 20,
                'placeholder': 'Number of bedrooms'
            }),
            'bathrooms': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0,
                'max': 20,
                'placeholder': 'Number of bathrooms'
            }),
            'house_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0,
                'step': '0.01',
                'placeholder': 'House size in m²'
            }),
            'land_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0,
                'step': '0.01',
                'placeholder': 'Land size in m²'
            }),
        }
        
        labels = {
            'property_type': 'Property Type',
            'location': 'Location',
            'bedrooms': 'Number of Bedrooms',
            'bathrooms': 'Number of Bathrooms',
            'house_size': 'House Size (m²)',
            'land_size': 'Land Size (m²)',
        }
        
        help_texts = {
            'house_size': 'Enter the built-up area in square meters',
            'land_size': 'Enter the plot size in square meters (1 acre = 4046.86 m²)',
        }
    
    def clean(self):
        cleaned_data = super().clean()
        house_size = cleaned_data.get('house_size')
        land_size = cleaned_data.get('land_size')
        bedrooms = cleaned_data.get('bedrooms')
        bathrooms = cleaned_data.get('bathrooms')
        
        
        if house_size == 0 and land_size == 0:
            raise forms.ValidationError(
                "Please provide either house size or land size"
            )
        
        
        if bedrooms > 0 and bathrooms == 0:
            self.add_error('bathrooms', 'Please specify number of bathrooms')
        
        return cleaned_data


class FeedbackForm(forms.ModelForm):
    """Form for user feedback on predictions"""
    
    class Meta:
        model = PropertyPrediction
        fields = ['user_rating', 'user_comments']
        
        widgets = {
            'user_rating': forms.RadioSelect(
                choices=[(i, i) for i in range(1, 6)],
                attrs={'class': 'rating-input'}
            ),
            'user_comments': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Share your feedback about this prediction...'
            }),
        }
        
        labels = {
            'user_rating': 'Rate this prediction',
            'user_comments': 'Additional comments (optional)',
        }


class ComparePropertiesForm(forms.Form):
    """Form to compare multiple properties"""
    
    property1_type = forms.ModelChoiceField(
        queryset=PropertyType.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Property Type 1'
    )
    property1_location = forms.ModelChoiceField(
        queryset=Location.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Location 1'
    )
    property1_bedrooms = forms.IntegerField(
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Bedrooms 1'
    )
    
    property2_type = forms.ModelChoiceField(
        queryset=PropertyType.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Property Type 2'
    )
    property2_location = forms.ModelChoiceField(
        queryset=Location.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Location 2'
    )
    property2_bedrooms = forms.IntegerField(
        min_value=0,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'}),
        label='Bedrooms 2'
    )
