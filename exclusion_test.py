import os
from typing import cast

import numpy as np
import cv2
import rasterio


FILENAME = "Test_Images/Test_Image_1.JPG"
IMAGE_BITS = 8

image_max_value = 2 ** IMAGE_BITS - 1
red_channel = 1
green_channel = 2
blue_channel = 3


def save_image(img: np.ndarray, image_name: str, *, dir: str = "./Test_Images"):
    """expects array input in BGR"""
    img_vals = (img * 255).astype(np.uint8)
    file_path = os.path.join(dir, f"{image_name}.jpg")
    cv2.imwrite(file_path, img_vals)


with rasterio.open(FILENAME, 'r') as raster2:
    raster2 = cast(rasterio.DatasetReader, raster2)

    blue_raw = cast(np.ndarray, raster2.read(blue_channel))
    green_raw = cast(np.ndarray, raster2.read(green_channel))
    red_raw = cast(np.ndarray, raster2.read(red_channel))

blue = blue_raw.astype(float) / image_max_value
green = green_raw.astype(float) / image_max_value
red = red_raw.astype(float) / image_max_value

np.seterr(divide='ignore', invalid='ignore')
NGRDI_Orig = (green - red) / (green + red)

save_image(
    np.where(
        np.repeat(np.expand_dims(NGRDI_Orig, 2), 3, axis=2) > 0,
        cv2.merge([blue, green, red]),
        0
    ),
    "exclusion_output"
)
