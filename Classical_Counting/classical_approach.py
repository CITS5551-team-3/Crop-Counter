import math
import time
from typing import cast, Optional

import cv2
import numpy as np
import rasterio


def count_image(filename: str, *, image_bits: int,
                red_channel = math.inf, green_channel = math.inf, blue_channel = math.inf,
                nir_channel = math.inf, re_channel = math.inf) -> int:
    """expects `np.seterr(divide='ignore', invalid='ignore')`"""

    # load the image

    blue_raw: Optional[np.ndarray] = None
    green_raw: Optional[np.ndarray] = None
    red_raw: Optional[np.ndarray] = None
    # nir_raw: Optional[np.ndarray] = None
    # re_raw: Optional[np.ndarray] = None

    # TODO should probably use cv2 to load the image
    with rasterio.open(filename, 'r') as raster_img:
        raster_img = cast(rasterio.DatasetReader, raster_img)
        band_count = cast(int, raster_img.count)

        if (band_count >= red_channel):
            red_raw = raster_img.read(red_channel)

        if (band_count >= green_channel):
            green_raw = raster_img.read(green_channel)

        if (band_count >= blue_channel):
            blue_raw = raster_img.read(blue_channel)

        # if (band_count >= nir_channel):
        #     nir_raw = raster_img.read(nir_channel)

        # if (band_count >= re_channel):
        #     re_raw = raster_img.read(re_channel)


    # convert from ints to 0-1 floats

    red: Optional[np.ndarray]
    green: Optional[np.ndarray]
    blue: Optional[np.ndarray]
    # nir: Optional[np.ndarray]
    # re: Optional[np.ndarray]

    image_max_value = 2 ** image_bits - 1

    if red_raw is not None:
        red = red_raw.astype(float) / image_max_value

    if green_raw is not None:
        green = green_raw.astype(float) / image_max_value

    if blue_raw is not None:
        blue = blue_raw.astype(float) / image_max_value

    # if nir_raw is not None:
    #     nir = nir_raw.astype(float) / image_max_value

    # if re_raw is not None:
    #     re = re_raw.astype(float) / image_max_value


    if red is None or green is None or blue is None:
        raise ValueError("not all rgb channels available")

    img = np.multiply(cv2.merge([blue, green, red]), 255).astype(np.uint8)


    # generate mask

    NGRDI = np.subtract(green, red) / np.add(green, red)
    NGRDI_mask = cv2.threshold(NGRDI, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)


    # mask and process the image

    img = cv2.bitwise_or(img, img, mask=NGRDI_mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    # image cleaning with erosion/dilation

    img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 8)
    img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 9)


    # count contours

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    def contours_filter(contour: cv2.typing.MatLike) -> bool:
        """True => keep"""

        # area = cv2.contourArea(contour)
        # if area < 600: return False
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter < 10: return False

        return True

    contours = [contour for contour in contours if contours_filter(contour)]

    crop_count = len(contours)
    return crop_count



def main():
    np.seterr(divide='ignore', invalid='ignore')

    IMAGE_DIR = "Classical_Counting/Test_Images"
    IMAGE_BITS = 8
    RED_CHANNEL = 1
    GREEN_CHANNEL = 2
    BLUE_CHANNEL = 3

    # file, manual count (or None)
    test_files: list[tuple[str, int | None]] = [
        (f"{IMAGE_DIR}/Test_Image_1.JPG", None),
        (f"{IMAGE_DIR}/Test_Image_2.JPG", None),
        (f"{IMAGE_DIR}/Test_Image_3.JPG", 397),
        (f"{IMAGE_DIR}/Test_Image_4.JPG", None),
        (f"{IMAGE_DIR}/Test_Image_5.JPG", None),
        (f"{IMAGE_DIR}/Test_Image_6.JPG", 324),
        (f"{IMAGE_DIR}/Test_Image_7.JPG", None),
    ]

    for file, manual_count in test_files:
        start_time = time.time()
        crop_count = count_image(
            file,
            image_bits=IMAGE_BITS,
            red_channel=RED_CHANNEL,
            green_channel=GREEN_CHANNEL,
            blue_channel=BLUE_CHANNEL
        )
        finish_time = time.time()

        count_duration_secs = finish_time - start_time

        accuracy_str = "unknown" if manual_count is None else f"{manual_count} - {crop_count/manual_count*100:.2f}%"
        print(f"Image: '{file}', Time Taken: {count_duration_secs:.5f} s, Count: {crop_count}/{accuracy_str}")

if __name__=="__main__":
    main()
