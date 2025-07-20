# Compressed-sensing

Demo of compressed sensing using `fmin_lbfgs` in Python.

## Installation
`pip install uv` if you don't have it already.
`uv sync` to install dependencies.

## Usage
1. Edit the parameters in `src/main.py`:
   - `image_size`: Size of the output image as `(width, height)`.
   - `sample_ratio`: Ratio of pixels to sample from the image.
   - `iterations`: Number of iterations for the optimization.
2. Run the script:
    - `uv run src/main.py`
3. A matplotlib window will open showing the input, samples, and reconstructed image.