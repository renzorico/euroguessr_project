import pandas as pd
import random
import geopandas as gpd
from ordered_set import OrderedSet

selected_coordinates = OrderedSet()  # Store selected coordinates

# Load land polygons shapefile excluding Antarctica
land_polygons = gpd.read_file("../raw_data/ne_10m_land.shp")

def is_land(lat, lon):
    point = gpd.points_from_xy([lon], [lat])
    for polygon in land_polygons.geometry:
        if polygon.contains(point[0]):
            return True
    return False

def generate_random_coordinates():
    min_lat, max_lat = -60, 75
    min_lon, max_lon = -180, 180

    while True:
        lat = round(random.uniform(min_lat, max_lat), 6)
        lon = round(random.uniform(min_lon, max_lon), 6)

        # Check if the coordinates are on land (excluding Antarctica)
        if is_land(lat, lon):
            coordinates = (lat, lon)

            if coordinates not in selected_coordinates:
                selected_coordinates.add(coordinates)
                return lat, lon
            
def read_coordinates_from_file(file_path):
    """
    Read coordinates from a text file and extract latitude and longitude.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    coordinates = []
    for line in lines:
        lat, lon = line.strip().split(',')
        latitude = float(lat)
        longitude = float(lon)
        coordinates.append((latitude, longitude))

    return coordinates

# import random

# selected_coordinates = set()  # Store selected coordinates

# def generate_random_coordinates():
#     min_lat, max_lat = 41.3429, 41.4765
#     min_lon, max_lon = 2.0899, 2.2280

#     while True:
#         lat = round(random.uniform(min_lat, max_lat), 5)
#         lon = round(random.uniform(min_lon, max_lon), 5)
#         coordinates = (lat, lon)

#         if coordinates not in selected_coordinates:
#             selected_coordinates.add(coordinates)
#             return lat, lon

# # Generate random coordinates within Barcelona city boundaries
# lat1, lon1 = generate_random_coordinates()

# print("First pair of coordinates:", lat1, lon1)
