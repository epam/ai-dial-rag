import base64
import io

import PIL.Image as pil_image


def image_as_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def pil_image_as_base64(image: pil_image.Image, format=None) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return image_as_base64(buffered.getvalue())


def pil_image_from_base64(image_base64: str) -> pil_image.Image:
    return pil_image.open(io.BytesIO(base64.b64decode(image_base64)))
