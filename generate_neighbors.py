import geopandas as gpd

# Load admin boundaries
admin1 = gpd.read_file('data/admin_bounds/eth_admin_boundaries.geojson/eth_admin1.geojson')
admin2 = gpd.read_file('data/admin_bounds/eth_admin_boundaries.geojson/eth_admin2.geojson')

# Get Benishangul boundary
bg = admin1[admin1['adm1_name'] == 'Benishangul Gumz']

# Find which regions border Benishangul
neighbors = []
for idx, region in admin1.iterrows():
    if region['adm1_name'] != 'Benishangul Gumz':
        if bg.geometry.iloc[0].touches(region.geometry) or bg.geometry.iloc[0].intersects(region.geometry):
            neighbors.append(region['adm1_name'])

print("Regions that border Benishangul Gumz:")
print(neighbors)

# Now check which zones are being wrongly assigned
wrong_zones = [
    ('Amhara', 'South Gondar'), ('Tigray', 'South Eastern'), ('Amhara', 'West Gojam'),
    ('Addis Ababa', 'Gulele'), ('Oromia', 'East Shewa'), ('SNNP', 'Wolayita'),
    ('South West Ethiopia', 'Kefa'), ('Addis Ababa', 'Kirkos'), ('Addis Ababa', 'Bole'),
    ('Somali', 'Korahe'), ('Sidama', 'Sidama'), ('Somali', 'Shabelle'),
    ('Gambela', 'Majang'), ('Harari', 'Abadir'), ('Oromia', 'Finfine Special'),
    ('Dire Dawa', 'Dire Dawa rural')
]

print("\nChecking if wrongly assigned zones are near Benishangul:")
for region_name, zone_name in wrong_zones:
    zone = admin2[(admin2['adm1_name'] == region_name) & (admin2['adm2_name'] == zone_name)]
    if not zone.empty:
        is_neighbor = bg.geometry.iloc[0].touches(zone.geometry.iloc[0]) or bg.geometry.iloc[0].distance(zone.geometry.iloc[0]) < 0.1
        dist = bg.geometry.iloc[0].distance(zone.geometry.iloc[0])
        print(f"  {region_name:20} / {zone_name:20} - Distance: {dist:.6f}")
