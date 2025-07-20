# does the actual compressed sensing reconstruction.
# following: https://humaticlabs.com/blog/compressed-sensing-python/

from PIL import Image
import numpy as np
import scipy.fftpack as spfft
from image_sampler import Arrayx5
from lbfgs import fmin_lbfgs
import scipy.ndimage
from tqdm import tqdm


def dct2(x: np.ndarray) -> np.ndarray:
    return spfft.dct(spfft.dct(x.T, norm="ortho", axis=0).T, norm="ortho", axis=0)


def idct2(x: np.ndarray) -> np.ndarray:
    return spfft.idct(spfft.idct(x.T, norm="ortho", axis=0).T, norm="ortho", axis=0)


def reconstruct_image(
    samples: Arrayx5, image_size: tuple[int, int], iterations_per_channel: int
) -> Image.Image:
    """
    Reconstruct an image from sampled pixels.

    Args:
        samples (Arrayx5): An array of shape (num_samples, 5) where each row contains
                           [x, y, R, G, B] for the sampled pixel.
        image_size (tuple[int, int]): The size of the output image as (width, height).
        iterations (int): Number of iterations per channel for the optimization.

    Returns:
        Image.Image: The reconstructed image.
    """

    width, height = image_size
    x_coords, y_coords = samples[:, 0], samples[:, 1]
    rgb_values = samples[:, 2:5]
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)
    with tqdm(
        total=3 * iterations_per_channel, desc="Reconstructing Image", unit="it"
    ) as pbar:
        for channel in range(3):
            b = rgb_values[:, channel]
            ri = (y_coords * width + x_coords).astype(int)

            def evaluate(x: np.ndarray, g: np.ndarray):
                x2 = x.reshape((width, height)).T
                Ax2 = idct2(x2)
                Ax = Ax2.T.flat[ri].reshape(b.shape)
                Axb = Ax - b
                fx = np.sum(np.power(Axb, 2))
                Axb2 = np.zeros(x2.shape)
                Axb2.T.flat[ri] = Axb
                AtAxb2 = 2 * dct2(Axb2)
                AtAxb = AtAxb2.T.reshape(x.shape)
                np.copyto(g, AtAxb)
                return fx

            def my_progress(x, g, fx, xnorm, gnorm, step, k, num_eval, *args):
                pbar.update(1)
                pbar.set_postfix(
                    fx=fx,
                    xnorm=xnorm,
                    gnorm=gnorm,
                    step=step,
                )
                return 0  # Continue; return non-zero to stop early

            x0 = np.zeros(width * height)
            x_reconstructed = fmin_lbfgs(
                evaluate,
                x0,
                orthantwise_c=1,
                progress=my_progress,
                max_iterations=iterations_per_channel,
                line_search="wolfe",
            )
            x_reconstructed_reshaped = x_reconstructed.reshape(width, height).T
            reconstructed_channel = idct2(x_reconstructed_reshaped)
            reconstructed_image[:, :, channel] = np.clip(
                reconstructed_channel, 0, 255
            ).astype(np.uint8)

    # transpose pixels to match the original image orientation
    reconstructed_image = np.transpose(reconstructed_image, (1, 0, 2))

    return Image.fromarray(reconstructed_image)
