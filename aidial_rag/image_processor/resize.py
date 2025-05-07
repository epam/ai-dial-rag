from contextlib import closing

import PIL.Image as pil_image

from aidial_rag.image_processor.base64 import (
    pil_image_as_base64,
    pil_image_from_base64,
)


def resize_image(img: pil_image.Image, size=(640, 640)):
    """
    Resize an image.

    Args:
    image (PIL.Image): Original image.
    size (tuple): Desired size of the image as (width, height).
    format (str): Format of the resized (e.g. "JPEG", "PNG").

    Returns:
    PIL.Image: Resized image.
    """
    # recalculate size keeping proportions
    if img.size[0] > img.size[1]:
        new_size = (size[0], int(size[0] * img.size[1] / img.size[0]))
    else:
        new_size = (int(size[1] * img.size[0] / img.size[1]), size[1])

    resized_img = img.resize(new_size, pil_image.Resampling.LANCZOS)
    return resized_img


def resize_base64_image(
    base64_string: str, size=(640, 640), format: str | None = None
):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).
    format (str): Format of the resized (e.g. "JPEG", "PNG").

    Returns:
    str: Base64 string of the resized image.
    """
    with closing(pil_image_from_base64(base64_string)) as img:
        resized_image = resize_image(img, size)
        return pil_image_as_base64(
            resized_image, format=(format if format else img.format)
        )
