from PIL import Image
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

# array of shape(N, 5)
Arrayx5: TypeAlias = NDArray[np.long]


def random_sample_pixels(image: Image.Image, num_pixels: int) -> Arrayx5:
    """
    Randomly sample pixels from the image and return their coordinates and RGB values.

    Args:
        image (Image.Image): The input image.
        num_pixels (int): The number of pixels to sample.

    Returns:
        Arrayx5: An array of shape (num_pixels, 5) where each row contains
                  [x, y, R, G, B] for the sampled pixel.
    """
    width, height = image.size
    pixels = np.array(image)

    # Randomly select pixel indices
    indices = np.random.choice(width * height, size=num_pixels, replace=False)

    # Convert flat indices to 2D coordinates
    x_coords = indices % width
    y_coords = indices // width

    # Get RGB values for the sampled pixels
    rgb_values = pixels[y_coords, x_coords]

    return np.column_stack((x_coords, y_coords, rgb_values))
