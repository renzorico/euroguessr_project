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
