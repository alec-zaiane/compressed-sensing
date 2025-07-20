import matplotlib.pyplot as plt

from image_fetcher import fetch_image
from image_sampler import random_sample_pixels
from compressed_sensing import reconstruct_image


def main():
    print("Fetching image...")
    image = fetch_image(200)
    print("Sampling pixels...")
    sample = random_sample_pixels(image, num_pixels=2000)
    print("Reconstructing image...")
    reconstructed_image = reconstruct_image(
        sample, image.size, iterations_per_channel=5000
    )

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].scatter(sample[:, 0], sample[:, 1], c=sample[:, 2:5] / 255.0, s=1)
    ax[1].set_title("Sampled Pixels")
    ax[1].invert_yaxis()  # Invert y-axis to match image coordinates
    ax[2].imshow(reconstructed_image)
    ax[2].set_title("Reconstructed Image")
    plt.show()


if __name__ == "__main__":
    main()
