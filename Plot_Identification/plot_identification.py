from typing import *

import numpy as np
from geotiff import GeoTiff
import cv2
from numba import jit
from skimage.measure import block_reduce

disable_showimg = True


def showimg(image, title="image"):
    if disable_showimg:
        return
    name = f"{title}_{image.shape}"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)


def filter_ExGI(rgb_image: np.ndarray) -> np.ndarray:
    """
    Apply ExGI filtering
    :param rgb_image: rgb image
    :return: grayscale image
    """
    R, G, B = (np.array(rgb_image[:, :, 0], dtype=np.dtype(int)),
               np.array(rgb_image[:, :, 1], dtype=np.dtype(int)),
               np.array(rgb_image[:, :, 2], dtype=np.dtype(int)))
    ExGI = (2 * G - R - B)
    ExGI_normalized = cv2.normalize(ExGI, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    ExGI_normalized = np.uint8(ExGI_normalized)
    return ExGI_normalized


def pad_image(image: np.ndarray, border_width: int):
    """
    Add paddings to an image.
    :param image:
    :param border_width: padding width in 4 directions
    :return:
    """
    return cv2.copyMakeBorder(
        image, top=border_width, bottom=border_width, left=border_width, right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )


def find_rectangles(bin_image, w_h_ratio, r_tolerance, exp_h, h_tolerance, annotate_on=None) -> List[Tuple[int, int, int, int]]:
    """
    Find rectangles by finding bounding rect of contour.
    :param bin_image:
    :param w_h_ratio:
    :param r_tolerance:
    :param exp_h:
    :param h_tolerance:
    :param annotate_on:
    :return:
    """
    colored = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2RGB) if annotate_on is None else annotate_on
    edged = cv2.Canny(bin_image, 30, 150)
    # showimg(edged)

    whr_lower_1 = w_h_ratio / (1 + r_tolerance)
    whr_upper_1 = w_h_ratio * (1 + r_tolerance)
    exp_h_lower = exp_h / (1 + h_tolerance)
    exp_h_upper = exp_h * (1 + h_tolerance)
    # show_img(edged)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(colored, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(colored, f"{w}-{h}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        cv2.drawContours(colored, [contour], 0, (0, 255, 0), 2)
        aspect_ratio = w / float(h)
        if h < exp_h_lower or h > exp_h_upper:
            continue
        if whr_lower_1 < aspect_ratio < whr_upper_1:
            rectangles.append((x, y, w, h))
            cv2.rectangle(colored, (x, y), (x + w, y + h), (255, 0, 0), 3)

    showimg(colored)
    return rectangles


@jit(nopython=True)
def calculate_span(image, ox, oy):
    """
    Find the topmost, leftmost, rightmost and bottommost continuous white pixel starting from (ox, oy)
    """
    minx, maxx, miny, maxy = ox, ox, oy, oy
    limx, limy = image.shape[1] - 1, image.shape[0] - 1
    while minx > 0 and image[oy, minx] > 0:
        minx -= 1
    while miny > 0 and image[miny, ox] > 0:
        miny -= 1
    while maxx < limx and image[oy, maxx] > 0:
        maxx += 1
    while maxy < limy and image[maxy, ox] > 0:
        maxy += 1
    return minx, miny, maxx, maxy


@jit(nopython=True)
def calculate_continuous_white_pixels(image):
    """
    Find the topmost, leftmost, rightmost and bottommost continuous white pixel
    """
    rows, cols = image.shape
    left = np.zeros_like(image, dtype=np.uint32)
    right = np.zeros_like(image, dtype=np.uint32)
    up = np.zeros_like(image, dtype=np.uint32)
    down = np.zeros_like(image, dtype=np.uint32)

    for i in range(rows):
        for j in range(cols):
            if image[i, j] > 0:
                left[i, j] = 1 if j == 0 else left[i, j - 1] + 1
                up[i, j] = 1 if i == 0 else up[i - 1, j] + 1

    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if image[i, j] > 0:
                right[i, j] = 1 if j == cols - 1 else right[i, j + 1] + 1
                down[i, j] = 1 if i == rows - 1 else down[i + 1, j] + 1

    return left, right, up, down


def calculate_span2(wL, wR, wU, wD, ox, oy):
    """
    Calculate the topmost, leftmost, rightmost and bottommost continuous white pixel starting from (ox, oy)
    using data obtained from calculate_continuous_white_pixels()
    """
    l, r, u, d = wL[oy, ox], wR[oy, ox], wU[oy, ox], wD[oy, ox]
    return ox-l, oy-u, ox+r, oy+d


def filter_rects(rects, w_h_ratio, r_tolerance, exp_h, h_tolerance):
    """
    Filter rectangles by aspect ratio and size, allowing some deviations
    :param rects:
    :param w_h_ratio:
    :param r_tolerance:
    :param exp_h:
    :param h_tolerance:
    :return:
    """
    whr_lower_1 = w_h_ratio / (1 + r_tolerance)
    whr_upper_1 = w_h_ratio * (1 + r_tolerance)
    exp_h_lower = exp_h / (1 + h_tolerance)
    exp_h_upper = exp_h * (1 + h_tolerance)

    result = []
    for x1, y1, x2, y2 in rects:
        spanw, spanh = x2 - x1, y2 - y1
        if spanw * spanh == 0:
            continue
        if spanh < exp_h_lower or spanh > exp_h_upper:
            continue
        aspect_ratio = spanw / float(spanh)
        if whr_lower_1 < aspect_ratio < whr_upper_1:
            result.append((x1, y1, x2, y2))
    return result


def find_rectangles2(bin_image, w_h_ratio, r_tolerance, exp_h, h_tolerance, crop_bleeding, annotate_on=None):
    """
    """
    whr_lower_1 = w_h_ratio / (1 + r_tolerance)
    whr_upper_1 = w_h_ratio * (1 + r_tolerance)
    exp_h_lower = exp_h / (1 + h_tolerance)
    exp_h_upper = exp_h * (1 + h_tolerance)

    c = 8
    step = max(1, min(exp_h * w_h_ratio // c, exp_h // c))
    expected_count = exp_h * exp_h * w_h_ratio / step / step
    # print(f"step={step}  tot={bin_image.shape[0] // c * bin_image.shape[1] // c}  expected={expected_count}")
    showimg(bin_image, "bin")

    # wL, wR, wU, wD = calculate_continuous_white_pixels(bin_image)

    counter = np.zeros(bin_image.shape)
    for y in range(0, bin_image.shape[0], step):
        for x in range(0, bin_image.shape[1], step):
            if bin_image[y, x] == 0:
                continue

            x1, y1, x2, y2 = calculate_span(bin_image, x, y)
            # x1, y1, x2, y2 = find_span2(wL, wR, wU, wD, x, y)

            spanw, spanh = x2 - x1, y2 - y1
            if spanw * spanh == 0:
                continue
            if spanh < exp_h_lower or spanh > exp_h_upper:
                continue
            aspect_ratio = spanw / float(spanh)
            if True or whr_lower_1 < aspect_ratio < whr_upper_1:
                counter[y1:y2 + 1, x1:x2 + 1] += 1
    # showimg(np.array(cv2.normalize(counter, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8), "rects")
    # print(f"levels: {np.unique(counter)}")
    counter = np.where(counter > expected_count / 12, 255, 0)  # TODO hardcoded
    counter = np.array(counter, dtype=np.uint8)
    showimg(counter, "counter")
    counter = remove_boundary_rects(counter, crop_bleeding)
    return find_rectangles(counter, w_h_ratio, r_tolerance, exp_h, h_tolerance, annotate_on)


def find_plots_preprocess(bin_img, plant_distance, max_weed_patch_size):
    pad_w = max(plant_distance, max_weed_patch_size) * 2 + 1  # TODO *2?
    bin_img = pad_image(bin_img, pad_w)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (plant_distance, plant_distance))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (max_weed_patch_size, max_weed_patch_size))
    s1 = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel1)
    s2 = cv2.morphologyEx(s1, cv2.MORPH_OPEN, kernel2)
    s3 = s2[pad_w:-pad_w, pad_w:-pad_w]
    return s3


def mask_vegetation(rgb_img: np.ndarray, exgi_bin: np.ndarray) -> np.ndarray:
    """
    Apply vegetation mask on the original image
    :param rgb_img: rgb image
    :param exgi_bin: binarised image
    :return: rgb image
    """
    mask_3d = cv2.cvtColor(exgi_bin, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(rgb_img, mask_3d)


def remove_boundary_rects_fill(img, fill_info):
    for label, coord in fill_info.items():
        cv2.floodFill(img, None, coord, 0)
    return img


@jit(nopython=True)
def remove_boundary_rects_detect(crop_bleeding, img, labels):
    h, w = img.shape
    crop_bleeding_L, crop_bleeding_R, crop_bleeding_T, crop_bleeding_B = crop_bleeding
    fill_info = {}
    for x in range(0, w):
        for y in range(0, crop_bleeding_T):
            if labels[y, x] > 0 and labels[y, x] not in fill_info:
                fill_info[labels[y, x]] = (x, y)
        for y in range(h - crop_bleeding_B, h):
            if labels[y, x] > 0 and labels[y, x] not in fill_info:
                fill_info[labels[y, x]] = (x, y)
    for y in range(0, h):
        for x in range(0, crop_bleeding_L):
            if labels[y, x] > 0 and labels[y, x] not in fill_info:
                fill_info[labels[y, x]] = (x, y)
        for x in range(w - crop_bleeding_R, w):
            if labels[y, x] > 0 and labels[y, x] not in fill_info:
                fill_info[labels[y, x]] = (x, y)
    return fill_info


# remove connected components that touches the bleeding area
def remove_boundary_rects(img, crop_bleeding):
    output_image = np.copy(img)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    fill_info = remove_boundary_rects_detect(crop_bleeding, img, labels)
    output_image = remove_boundary_rects_fill(output_image, fill_info)
    return output_image


def find_plots(rgb_img, exp_w, exp_h, r_tolerance=0.15, h_tolerance=0.2, shrink_factor: int = 4,
               result_padding: float = 0.05, crop_bleeding=(0, 0, 0, 0), write_to_file=False):
    exgi = filter_ExGI(rgb_img)

    small_exgi = block_reduce(exgi, block_size=(shrink_factor, shrink_factor), func=np.max)

    # showimg(small_exgi, "ExGI")
    _, binarised = cv2.threshold(small_exgi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # showimg(binarised, "ExGI_binarised")

    # Output masked image
    # small_rgb = cv2.resize(rgb_img, (small_exgi.shape[1], small_exgi.shape[0]))
    # small_rgb = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2BGR)
    # aaa = mask_vegetation(small_rgb, binarised)
    # cv2.imwrite("data/masked.png", aaa)

    img = find_plots_preprocess(binarised, plant_distance=45, max_weed_patch_size=50)  # TODO hardcoded

    exp_w //= shrink_factor
    exp_h //= shrink_factor
    crop_bleeding = np.array(crop_bleeding) // 4
    bg = cv2.resize(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), dsize=(small_exgi.shape[1], small_exgi.shape[0]))
    rects = find_rectangles2(img, exp_w / exp_h, r_tolerance, exp_h, h_tolerance, crop_bleeding, bg)

    result = []

    index = 0
    padding_x, padding_y = int(exp_w * result_padding), int(exp_h * result_padding)
    for x, y, w, h in rects:
        x *= shrink_factor
        y *= shrink_factor
        w *= shrink_factor
        h *= shrink_factor
        minx = max(0, x-padding_x)
        maxx = min(rgb_img.shape[1], x + w + padding_x)
        miny = max(0, y-padding_y)
        maxy = min(rgb_img.shape[0], y + h + padding_y)
        result.append((minx, miny, maxx, maxy))

        if write_to_file:
            plot_img = rgb_img[miny:maxy, minx:maxx]
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/output_plots/plot_{index}.png", plot_img)
            index = index + 1
    return result


def merge_rects(rectangles: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.01):
    """
    Merge rectangles with overlapping areas more than overlap_theshold of total are of two rectangles.
    :param rectangles:
    :param overlap_threshold:
    :return:
    """
    rectangles = list(rectangles)

    def area(x1, y1, x2, y2):
        """Calculate the area of a rectangle."""
        return max(0, x2 - x1) * max(0, y2 - y1)

    def intersect(r1, r2):
        """Calculate the intersection rectangle of two rectangles."""
        x1 = max(r1[0], r2[0])
        y1 = max(r1[1], r2[1])
        x2 = min(r1[2], r2[2])
        y2 = min(r1[3], r2[3])
        if x1 < x2 and y1 < y2:
            return x1, y1, x2, y2
        return None

    while True:
        merged = False
        new_rectangles = []
        while len(rectangles) > 0:
            rect = rectangles.pop(-1)
            i = 0
            while i < len(rectangles):
                other = rectangles[i]
                intersection = intersect(rect, other)
                if not intersection:
                    i += 1
                    continue
                area_int = area(*intersection)
                area_rect = area(*rect)
                area_other = area(*other)
                if area_int > overlap_threshold * (area_rect + area_other - area_int):
                    rect = (min(rect[0], other[0]), min(rect[1], other[1]),
                            max(rect[2], other[2]), max(rect[3], other[3]))
                    rectangles.pop(i)
                    merged = True
                else:
                    i += 1
            new_rectangles.append(rect)
        rectangles = new_rectangles
        if not merged:
            break
    return rectangles


def find_plots_geotiff(tiff_path: str, exp_w, exp_h, r_tolerance=0.15, h_tolerance=0.2,
                       shrink_factor: int = 4, result_padding: float = 0.05, write_to_file=False):
    """
    Find plots from geoTIFF file. The image can be much larger than the memory can accommodate.
    :param tiff_path: path to geoTIFF file
    :param exp_w: expected plot width (in pixels)
    :param exp_h: expected plot height (in pixels)
    :param r_tolerance:
    :param h_tolerance:
    :param shrink_factor: shrink the image by shrink_factor before processing
    :param result_padding:
    :param write_to_file:
    :return:
    """
    geo_tiff = GeoTiff(tiff_path)
    zarr = geo_tiff.read()

    zarr_width, zarr_height = zarr.shape[1], zarr.shape[0]
    batch_w, batch_h = exp_w * 2, exp_h * 2  # increase/decrease this according to available memory
    print(f"expected plot size: {exp_w}x{exp_h}    "
          f"image size: {zarr_width}x{zarr_height}   "
          f"single detection size: {batch_w}x{batch_h}")

    extra_x, extra_y = int(exp_w * (1 + h_tolerance)) + 1, int(exp_h * (1 + h_tolerance)) + 1
    bleeding = 0.1
    bleeding_x, bleeding_y = int(exp_w * bleeding) + 1, int(exp_h * bleeding) + 1

    plots = []
    rgb_img = np.array(zarr)[:, :, :3]  # TODO for testing, remove this when dealing with large images
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    for y1 in range(0, zarr_height, batch_h):
        for x1 in range(0, zarr_width, batch_w):
            x2 = min(zarr_width, x1 + batch_w + extra_x)
            y2 = min(zarr_height, y1 + batch_h + extra_y)
            crop_x1 = max(0, x1 - bleeding_x)
            crop_y1 = max(0, y1 - bleeding_y)
            crop_x2 = min(zarr_width, x2 + bleeding_x + 1)
            crop_y2 = min(zarr_height, y2 + bleeding_y + 1)
            crop_bleeding_L = x1 - crop_x1
            crop_bleeding_R = crop_x2 - x2
            crop_bleeding_T = y1 - crop_y1
            crop_bleeding_B = crop_y2 - y2
            crop_bleeding = (crop_bleeding_L, crop_bleeding_R, crop_bleeding_T, crop_bleeding_B)
            # print(f"bleeding = {crop_bleeding}")

            img = zarr[crop_y1:crop_y2, crop_x1:crop_x2, :3]
            # cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color=(np.random.randint(0,255),np.random.randint(0,255),255),thickness=3)

            tmp_plots = find_plots(img, exp_w, exp_h, r_tolerance, h_tolerance, shrink_factor, result_padding, crop_bleeding)
            for u1, v1, u2, v2 in tmp_plots:
                coord = (crop_x1+u1, crop_y1+v1, crop_x1+u2, crop_y1+v2)
                print(f"Found plot: {coord}")
                plots.append(coord)

    plots = np.unique(plots, axis=0)
    print(plots)
    plots = merge_rects(plots, 0)
    print(f"merged.  Found {len(plots)} plots")
    plots = filter_rects(plots, exp_w / exp_h, r_tolerance, exp_h, h_tolerance)
    print(f"filtered.  Found {len(plots)} plots")
    print(plots)

    # TODO for testing, remove this in the future
    for x1, y1, x2, y2 in plots:
        color = (np.random.randint(0, 255), np.random.randint(0, 255), 255)
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), color=color,thickness=3)
    showimg(rgb_img)
    cv2.imwrite("data/plots_result.png", rgb_img)

    # write results
    if write_to_file:
        index = 0
        for x1, y1, x2, y2 in plots:
            plot_img = zarr[y1:y2, x1:x2]
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"data/output_plots/plot_{index}.png", plot_img)
            index = index + 1

    return plots

# TODO: comment the code

# Plot identification is done by applying ExGI filtering, binarisation, and fitting rectangles into the binarised image.
# This method currently has a limitation: the rectangles must be parallel to the image boundaries.
# The GeoTIFF file can potentially be several gigabytes which is too large to fit into the memory. But fortunately
# the geotiff package has done the memory management for us, allowing access to only a small portion of the image
# at a time without needing to load the entire image. This method takes advantage of this feature.
# It detects plots within small areas, and then combines the results at the end.


def main():
    tiff_file = "data/plots.tif"  # https://drive.google.com/file/d/1Hnz7eY6rajJGGptk3Lw9NWmL5W6uf50Z/view?usp=sharing
    tiff_file = "data/plots_with_weed.tif"  # https://drive.google.com/file/d/1CUoaFNcektBEsoaz_LDVz6akUnrqgvBL/view?usp=sharing
    expected_plot_width, expected_plot_height = 3280, 600  # 420/150,1100/150
    arr = cv2.imread("data/plots2.png")

    rgb_img = np.array(arr)[:, :, :3]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # find_plots(rgb_img, expected_plot_width=expected_plot_width, expected_plot_height=expected_plot_height, write_to_file=True)
    find_plots_geotiff(tiff_file, exp_w=expected_plot_width, exp_h=expected_plot_height, write_to_file=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
