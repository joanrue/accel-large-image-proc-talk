# +

import itertools

# Chunk image and save chunk
import os
from pathlib import Path

import numpy as np
import skimage

import matplotlib.pyplot as plt
import numba
# -

__all__ = ["show_images", "apply_kernel_x", "apply_kernel_y"]

def show_images(images, zoom=False, titles=None, cmap="gray", figsize=(20,10)):
    zoom_mask = (slice(200, 800), slice(2320, 2750))
    if cmap is None or isinstance(cmap, str):
        cmap = [cmap, ] * len(images)
    if titles is None or isinstance(titles, str):
        titles = [titles, ] * len(images)
    if zoom is None or isinstance(zoom, bool):
        zoom = [zoom, ] * len(images)
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    for ax, image in enumerate(images):
        if zoom[ax]:
            axs[ax].imshow(image[zoom_mask], cmap=cmap[ax])
            axs[ax].set_title(titles[ax] + "(zoom)")
        else:
            axs[ax].imshow(image, cmap=cmap[ax])
            axs[ax].set_title(titles[ax])
        axs[ax].axis("off")
    fig.tight_layout()


@numba.stencil(neighborhood=((0, 0), (-10, 10), (0, 0)), standard_indexing=("kernel",))
def apply_kernel_x(images, kernel):
    """
    Apply a kernel on the x-axis of a 3D image.
    
    Parameters
    ----------
        images: nd-array (npix_y, npix_x, nchan)
                Stack of images
                
    Returns
    -------
        filtered_image: nd-array (npix_y, npix_x, nchan)
                Stack of images
    """

    return (
        images[0, -10, 0] * kernel[0]
        + images[0, -9, 0] * kernel[1]
        + images[0, -8, 0] * kernel[2]
        + images[0, -7, 0] * kernel[3]
        + images[0, -6, 0] * kernel[4]
        + images[0, -5, 0] * kernel[5]
        + images[0, -4, 0] * kernel[6]
        + images[0, -3, 0] * kernel[7]
        + images[0, -2, 0] * kernel[8]
        + images[0, -1, 0] * kernel[9]
        + images[0, 0, 0] * kernel[10]
        + images[0, 1, 0] * kernel[11]
        + images[0, 2, 0] * kernel[12]
        + images[0, 3, 0] * kernel[13]
        + images[0, 4, 0] * kernel[14]
        + images[0, 5, 0] * kernel[15]
        + images[0, 6, 0] * kernel[16]
        + images[0, 7, 0] * kernel[17]
        + images[0, 8, 0] * kernel[18]
        + images[0, 9, 0] * kernel[19]
        + images[0, 10, 0] * kernel[20]
    )


@numba.stencil(neighborhood=((-10, 10), (0, 0), (0, 0)), standard_indexing=("kernel",))
def apply_kernel_y(images, kernel):
    """
    Apply a kernel on the y-axis of a 3D image.
    
    Parameters
    ----------
        images: nd-array (npix_y, npix_x, nchan)
                Stack of images
                
    Returns
    -------
        filtered_image: nd-array (npix_y, npix_x, nchan)
                Stack of images
    """
    return (
          images[-10, 0, 0] * kernel[0]
        + images[ -9, 0, 0] * kernel[1]
        + images[ -8, 0, 0] * kernel[2]
        + images[ -7, 0, 0] * kernel[3]
        + images[ -6, 0, 0] * kernel[4]
        + images[ -5, 0, 0] * kernel[5]
        + images[ -4, 0, 0] * kernel[6]
        + images[ -3, 0, 0] * kernel[7]
        + images[ -2, 0, 0] * kernel[8]
        + images[ -1, 0, 0] * kernel[9]
        + images[  0, 0, 0] * kernel[10]
        + images[  1, 0, 0] * kernel[11]
        + images[  2, 0, 0] * kernel[12]
        + images[  3, 0, 0] * kernel[13]
        + images[  4, 0, 0] * kernel[14]
        + images[  5, 0, 0] * kernel[15]
        + images[  6, 0, 0] * kernel[16]
        + images[  7, 0, 0] * kernel[17]
        + images[  8, 0, 0] * kernel[18]
        + images[  9, 0, 0] * kernel[19]
        + images[ 10, 0, 0] * kernel[20]
    ) 


def chunk_image(image, chunk_size, output_dir):
    shape = np.array(image.shape)
    chunk = np.array(chunk_size)
    chunk_counts = shape // chunk
    for ids in itertools.product(*[np.arange(0, count) for count in chunk_counts]):
        slices = tuple([slice(i * ch, (i + 1) * ch) for i, ch in zip(ids, chunk)])
        string = "-".join([str(i).zfill(2) for i in ids])
        skimage.io.imsave(
            os.path.join(output_dir, f"image-{string}.png"), (image[slices] * 255).astype(np.uint8),
            check_contrast=False
        )

def list_files(startpath, max_len=10):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files[:max_len]:
            print(f'{subindent}{f} {os.path.getsize(os.path.join(root, f)) * 1e-3 : 0.2f} Kb')