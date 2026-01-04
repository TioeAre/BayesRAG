import base64, os
import io
from io import BytesIO

import numpy as np
from PIL import Image
from langchain_core.documents import Document
from loguru import logger


def resize_image(image, size=(256, 256)):
    img = Image.open(image)
    resized_img = img.resize(size, Image.LANCZOS)  # type: ignore
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def resize_base64_image(base64_string, size=(256, 256)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)  # type: ignore
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs: list[Document]):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content
        if is_base64(doc):
            images.append(resize_base64_image(doc, size=(250, 250)))
        else:
            text.append(doc)
    return {"images": images, "texts": text}


def save_base64_image(base64_string: str, output_path: str):
    """
    Decodes a Base64 string and saves it as an image file.

    Args:
        base64_string: base64 encoded string of the image.
        output_path: the full path to be saved.
    """
    try:
        image_data = base64.b64decode(base64_string)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as image_file:
            image_file.write(image_data)

        logger.info(f"Image successfully saved to: {output_path}")

    except base64.binascii.Error as e:  # type: ignore
        logger.error(f"Error decoding Base64 string: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def image2base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode("utf-8")
