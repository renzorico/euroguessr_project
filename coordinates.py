import random

selected_coordinates = set()  # Store selected coordinates

def generate_random_coordinates():
    min_lat, max_lat = -90, 90
    min_lon, max_lon = -180, 180

    while True:
        lat = random.uniform(min_lat, max_lat)
        lon = random.uniform(min_lon, max_lon)
        coordinates = (lat, lon)

        if coordinates not in selected_coordinates:
            selected_coordinates.add(coordinates)
            return lat, lon

# Generate random coordinates
lat1, lon1 = generate_random_coordinates()
lat2, lon2 = generate_random_coordinates()

print("First pair of coordinates:", lat1, lon1)
print("Second pair of coordinates:", lat2, lon2)
