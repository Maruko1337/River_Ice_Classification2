import ee
import numpy as np
import matplotlib.pyplot as plt

# Authenticate Earth Engine using your credentials
ee.Authenticate()

# Initialize Earth Engine
ee.Initialize(project="ee-quyf0516")

# Define the longitude, latitude, and date
lon, lat = -74.1691114199999930,45.3098480999999964
date_start = '2017-01-01'
date_end = '2017-01-31'

# Create a point geometry
point = ee.Geometry.Point(lon, lat)

# Define the Landsat image collection and filter by date and location
landsat_collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA') \
    .filterBounds(point) \
    .filterDate(date_start, date_end)

# Select the first image from the collection
landsat_image = landsat_collection.first()
# Check if landsat_image is not None
if landsat_image is not None:
    # Get the visualization parameters for Landsat imagery (RGB bands)
    vis_params = {
        'bands': ['B4', 'B3', 'B2'],  # True color (Red, Green, Blue)
        'min': 0,
        'max': 0.3
    }

    # # Get the thumbnail URL for visualization
    # thumbnail_url = landsat_image.getThumbURL(vis_params)
    
    # Get the map ID and token
    map_id = landsat_image.getMapId(vis_params)

    # Construct the thumbnail URL manually
    thumbnail_url = 'https://earthengine.googleapis.com/v1alpha/projects/{ee-quyf0516}/thumb/'.format(ee.data._private.project_id)
    thumbnail_url += map_id['mapid'] + '?token=' + map_id['token']


    print("Thumbnail URL:", thumbnail_url)
else:
    print("No Landsat image found in the collection.")

# Load the image into a NumPy array
image_array = plt.imread(thumbnail_url)

# Plot the image
plt.imshow(image_array)
plt.title('Landsat Imagery')
plt.axis('off')
plt.savefig("googleEarthImage.jpg")