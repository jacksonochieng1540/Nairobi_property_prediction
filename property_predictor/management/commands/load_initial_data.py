# nairobi_property_predictor/property_predictor/management/commands/load_initial_data.py

from django.core.management.base import BaseCommand
from property_predictor.models import Location, PropertyType


class Command(BaseCommand):
    help = 'Load initial locations and property types'

    def handle(self, *args, **kwargs):
        self.stdout.write('Loading initial data...')
        
        # Load locations with tiers
        locations_data = [
            # Tier 1 - Premium
            {'name': 'Runda', 'tier': 'Tier1_Premium'},
            {'name': 'Muthaiga', 'tier': 'Tier1_Premium'},
            {'name': 'Muthaiga North', 'tier': 'Tier1_Premium'},
            {'name': 'Karen', 'tier': 'Tier1_Premium'},
            {'name': 'Kitisuru', 'tier': 'Tier1_Premium'},
            
            # Tier 2 - Upscale
            {'name': 'Lavington', 'tier': 'Tier2_Upscale'},
            {'name': 'Kileleshwa', 'tier': 'Tier2_Upscale'},
            {'name': 'Westlands', 'tier': 'Tier2_Upscale'},
            {'name': 'Nyari', 'tier': 'Tier2_Upscale'},
            {'name': 'Loresho', 'tier': 'Tier2_Upscale'},
            {'name': 'Rosslyn', 'tier': 'Tier2_Upscale'},
            {'name': 'Thigiri', 'tier': 'Tier2_Upscale'},
            {'name': 'Riverside', 'tier': 'Tier2_Upscale'},
            
            # Tier 3 - Standard
            {'name': 'Kilimani', 'tier': 'Tier3_Standard'},
            {'name': 'Parklands', 'tier': 'Tier3_Standard'},
            {'name': 'Kyuna', 'tier': 'Tier3_Standard'},
            {'name': 'Kabete', 'tier': 'Tier3_Standard'},
            {'name': 'Lower Kabete', 'tier': 'Tier3_Standard'},
            {'name': 'Kiambu Road', 'tier': 'Tier3_Standard'},
            {'name': 'Ngong Rd', 'tier': 'Tier3_Standard'},
            {'name': 'Nairobi West', 'tier': 'Tier3_Standard'},
            {'name': 'Syokimau', 'tier': 'Tier3_Standard'},
            {'name': 'Thome', 'tier': 'Tier3_Standard'},
            {'name': 'Ongata Rongai', 'tier': 'Tier3_Standard'},
            {'name': 'Waithaka', 'tier': 'Tier3_Standard'},
        ]
        
        location_count = 0
        for loc_data in locations_data:
            location, created = Location.objects.get_or_create(
                name=loc_data['name'],
                defaults={'tier': loc_data['tier']}
            )
            if created:
                location_count += 1
                self.stdout.write(f'  Created location: {location.name}')
        
        self.stdout.write(self.style.SUCCESS(
            f'✓ Loaded {location_count} new locations'
        ))
        
        # Load property types
        property_types = [
            {'name': 'Apartment', 'description': 'Multi-unit residential building'},
            {'name': 'Townhouse', 'description': 'Multi-floor house in a residential complex'},
            {'name': 'Vacant Land', 'description': 'Undeveloped land plot'},
            {'name': 'Commercial Property', 'description': 'Property for business use'},
            {'name': 'Industrial Property', 'description': 'Property for industrial use'},
        ]
        
        type_count = 0
        for pt_data in property_types:
            prop_type, created = PropertyType.objects.get_or_create(
                name=pt_data['name'],
                defaults={'description': pt_data['description']}
            )
            if created:
                type_count += 1
                self.stdout.write(f'  Created property type: {prop_type.name}')
        
        self.stdout.write(self.style.SUCCESS(
            f'✓ Loaded {type_count} new property types'
        ))
        
        self.stdout.write(self.style.SUCCESS(
            '\n✓ Initial data loaded successfully!'
        ))