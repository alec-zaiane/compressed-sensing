# Fetches an image from https://picsum.photos/

import requests
from PIL import Image
from io import BytesIO


# https://stackoverflow.com/questions/7391945/how-do-i-read-image-data-from-a-url-in-python
def fetch_image(size: int):
    res = requests.get(f"https://picsum.photos/{size}")
    return Image.open(BytesIO(res.content))
