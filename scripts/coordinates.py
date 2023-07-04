# import random

# selected_coordinates = set()  # Store selected coordinates

# def generate_random_coordinates():
#     min_lat, max_lat = -60,60
#     min_lon, max_lon = -180, 180

#     while True:
#         lat = round(random.uniform(min_lat, max_lat),5)
#         lon = round(random.uniform(min_lon, max_lon),5)
#         coordinates = (lat, lon)

#         if coordinates not in selected_coordinates:
#             selected_coordinates.add(coordinates)
#             return lat, lon

# # Generate random coordinates
# lat1, lon1 = generate_random_coordinates()

# print("First pair of coordinates:", lat1, lon1)


# import random

# selected_coordinates = set()  # Store selected coordinates

# def generate_random_coordinates():
#     min_lat, max_lat = 36.0000, 71.0000
#     min_lon, max_lon = -31.0000, 42.0000

#     while True:
#         lat = round(random.uniform(min_lat, max_lat), 5)
#         lon = round(random.uniform(min_lon, max_lon), 5)
#         coordinates = (lat, lon)

#         if coordinates not in selected_coordinates:
#             selected_coordinates.add(coordinates)
#             return lat, lon

# # Generate random coordinates within Europe
# lat1, lon1 = generate_random_coordinates()

# print("First pair of coordinates:", lat1, lon1)

import random

selected_coordinates = set()  # Store selected coordinates

def generate_random_coordinates():
    min_lat, max_lat = 41.3429, 41.4765
    min_lon, max_lon = 2.0899, 2.2280

    while True:
        lat = round(random.uniform(min_lat, max_lat), 5)
        lon = round(random.uniform(min_lon, max_lon), 5)
        coordinates = (lat, lon)

        if coordinates not in selected_coordinates:
            selected_coordinates.add(coordinates)
            return lat, lon

# Generate random coordinates within Barcelona city boundaries
lat1, lon1 = generate_random_coordinates()

print("First pair of coordinates:", lat1, lon1)
