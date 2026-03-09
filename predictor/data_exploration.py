import pandas as pd
import json
import plotly.express as px
from django.conf import settings

# Data Exploration
def dataset_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
        index=False,
    )
    return table_html

# Data description
def data_exploration(df):
    table_html = df.head().to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        justify="center",
    )
    return table_html

# Exercise 19a - Plotly Map of Rwanda Districts
def generate_rwanda_map(df):
    # Group by district to get client counts
    district_counts = df.groupby('district').size().reset_index(name='client_count')
    
    # Load Rwanda geojson (we'll fetch a placeholder if not available, 
    # but for simple demonstration without downloading large valid geojsons during lab 
    # we can just create a basic scatter_mapbox or choropleth with built-ins, 
    # although proper choropleth needs geojson. We will use a scatter mapbox on approximate coordinates)
    
    # Approximation of district coordinates for Rwanda
    rwanda_districts_coords = {
        'Nyarugenge': (-1.95, 30.05), 'Gasabo': (-1.88, 30.13), 'Kicukiro': (-1.98, 30.11), # Kigali
        'Kamonyi': (-2.00, 29.88), 'Muhanga': (-2.08, 29.75), 'Ruhango': (-2.22, 29.77), 'Nyanza': (-2.35, 29.75), 'Huye': (-2.60, 29.75), 'Nyaruguru': (-2.72, 29.53), 'Gisagara': (-2.62, 29.83), # South
        'Karongi': (-2.15, 29.35), 'Rutsiro': (-1.93, 29.32), 'Rubavu': (-1.68, 29.25), 'Nyabihu': (-1.60, 29.50), 'Ngororero': (-1.87, 29.53), 'Rusizi': (-2.48, 28.90), 'Nyamasheke': (-2.35, 29.13), # West
        'Rulindo': (-1.77, 29.98), 'Gakenke': (-1.70, 29.78), 'Musanze': (-1.50, 29.63), 'Burera': (-1.43, 29.80), 'Gicumbi': (-1.62, 30.08), # North
        'Rwamagana': (-1.95, 30.43), 'Nyagatare': (-1.30, 30.32), 'Gatsibo': (-1.60, 30.45), 'Kayonza': (-1.92, 30.65), 'Kirehe': (-2.27, 30.65), 'Ngoma': (-2.17, 30.47), 'Bugesera': (-2.22, 30.15), # East
    }
    
    district_counts['lat'] = district_counts['district'].map(lambda x: rwanda_districts_coords.get(x, (0,0))[0])
    district_counts['lon'] = district_counts['district'].map(lambda x: rwanda_districts_coords.get(x, (0,0))[1])
    
    # Filter out anything with 0,0 (not found)
    district_counts_filtered = district_counts[district_counts['lat'] != 0]

    fig = px.scatter_mapbox(
        district_counts_filtered, 
        lat="lat", 
        lon="lon",     
        hover_name="district", 
        hover_data=["client_count"],
        color="client_count",
        size="client_count",
        color_continuous_scale=px.colors.cyclical.IceFire, 
        size_max=15, 
        zoom=7, 
        center={"lat": -1.94, "lon": 29.87},
        mapbox_style="carto-positron",
        title="Number of Vehicle Clients in Rwanda Districts"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
